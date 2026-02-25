"""
Channel Clusterer — K-Means clustering trên 42 kênh.
Dùng để assign cluster khi có kênh mới và tạo prior probability.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TRAINED_MODELS_DIR = Path(__file__).parent.parent.parent / "trained_models"
CLUSTERER_PATH = TRAINED_MODELS_DIR / "channel_clusterer.pkl"

CLUSTER_NAMES = {
    0: "Tier 0 — Mega Channels",
    1: "Tier 1 — Large Channels",
    2: "Tier 2 — Growing Channels",
    3: "Tier 3 — Niche Channels",
    4: "Tier 4 — Small Channels",
}


class ChannelClusterer:
    """
    K-Means clustering cho kênh YouTube.

    Usage:
        clusterer = ChannelClusterer()
        clusterer.fit(channel_features_df)  # sau khi có labels
        cluster_id, distance = clusterer.assign_cluster(new_channel_features)
        stats = clusterer.get_cluster_stats(cluster_id)
        ChannelClusterer.load() để load model đã train
    """

    CLUSTER_FEATURES = [
        "f9_sub_tier",          # log10(subscribers)
        "f6_avg_views",         # avg views per video → log transform
        "f2_loyalty",           # like/view ratio
        "f3_depth",             # comment/view ratio
    ]

    def __init__(self, n_clusters: int = 4) -> None:
        self.n_clusters = n_clusters
        self._scaler = StandardScaler()
        self._kmeans: Optional[KMeans] = None
        self._cluster_stats: dict[int, dict] = {}
        self._is_fitted = False

    # ── Fit ────────────────────────────────────────────────────────────────────
    def find_optimal_k(
        self,
        channel_features: pd.DataFrame,
        k_range: range = range(2, 9),
    ) -> int:
        """
        Tìm k tối ưu bằng Elbow Method + Silhouette Score.

        Returns:
            Optimal k (int)
        """
        X = self._extract_cluster_features(channel_features)
        X_scaled = self._scaler.fit_transform(X)

        inertias = []
        silhouettes = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            if k > 1:
                sil = silhouette_score(X_scaled, labels)
                silhouettes.append(sil)
            else:
                silhouettes.append(0.0)

        print(f"\n{'─'*50}")
        print(f"ELBOW METHOD + SILHOUETTE SCORE")
        print(f"{'─'*50}")
        for i, k in enumerate(k_range):
            sil = silhouettes[i]
            inert = inertias[i]
            bar = "█" * int(sil * 40)
            print(f"  k={k}: inertia={inert:8.0f}, silhouette={sil:.3f} {bar}")

        # Chọn k có silhouette cao nhất
        best_k_idx = int(np.argmax(silhouettes))
        best_k = list(k_range)[best_k_idx]
        print(f"\n    Optimal k = {best_k} (silhouette = {silhouettes[best_k_idx]:.3f})")
        print(f"{'─'*50}\n")
        return best_k

    def fit(
        self,
        channel_features: pd.DataFrame,
        labeled_df: Optional[pd.DataFrame] = None,
        auto_find_k: bool = True,
    ) -> "ChannelClusterer":
        """
        Fit K-Means trên channel features.

        Args:
            channel_features: DataFrame từ ChannelFeatureEngineer.transform()
            labeled_df:       DataFrame với cột is_viral_channel (để tính viral rate per cluster)
            auto_find_k:      True → tự tìm k tối ưu
        """
        X = self._extract_cluster_features(channel_features)

        if auto_find_k:
            k = self.find_optimal_k(channel_features, k_range=range(2, min(8, len(X))))
            self.n_clusters = k

        X_scaled = self._scaler.fit_transform(X)
        self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=20)
        cluster_labels = self._kmeans.fit_predict(X_scaled)

        channel_features = channel_features.copy()
        channel_features["cluster_id"] = cluster_labels

        # Merge viral labels nếu có
        if labeled_df is not None and "is_viral_channel" in labeled_df.columns:
            label_col = labeled_df["is_viral_channel"].values
            channel_features["is_viral_channel"] = label_col

        # Tính cluster stats
        self._compute_cluster_stats(channel_features)
        self._is_fitted = True

        self._print_cluster_summary(channel_features)
        logger.info("ChannelClusterer fitted: k=%d", self.n_clusters)
        return self

    def _compute_cluster_stats(self, df: pd.DataFrame) -> None:
        # Bước 1: tính stats thô
        raw: dict[int, dict] = {}

        # Tất cả numeric features cần lưu percentile
        PERCENTILE_FEATURES = [
            "f1_efficiency", "f2_loyalty", "f3_depth", "f4_consistency",
            "f6_avg_views", "f7_engagement", "f9_sub_tier", "f11_recent_trend",
        ]

        for cluster_id in range(self.n_clusters):
            mask = df["cluster_id"] == cluster_id
            subset = df[mask]
            stats: dict = {"count": int(mask.sum())}
            if "is_viral_channel" in subset.columns:
                stats["viral_rate"] = float(subset["is_viral_channel"].mean())
            else:
                stats["viral_rate"] = 0.5
            for feat in self.CLUSTER_FEATURES:
                if feat in subset.columns:
                    stats[f"median_{feat}"] = float(subset[feat].median())

            # Lưu per-feature percentiles để dùng lúc inference (within-cluster benchmark)
            for feat in PERCENTILE_FEATURES:
                if feat in subset.columns and len(subset) >= 3:
                    vals = subset[feat].dropna()
                    stats[f"p25_{feat}"]  = float(vals.quantile(0.25))
                    stats[f"p50_{feat}"]  = float(vals.quantile(0.50))
                    stats[f"p75_{feat}"]  = float(vals.quantile(0.75))
                    stats[f"min_{feat}"]  = float(vals.min())
                    stats[f"max_{feat}"]  = float(vals.max())

            raw[cluster_id] = stats

        # Bước 2: đặt tên động theo median f9_sub_tier (subscribers)
        # Cluster có sub tier cao nhất = Mega, tiếp theo Large, Growing, Niche…
        TIER_NAMES = [
            "Mega Channels (100M+ sub)",
            "Large Channels (1M–100M sub)",
            "Growing Channels (100K–1M sub)",
            "Niche Channels (<100K sub)",
            "Micro Channels",
        ]
        sorted_ids = sorted(
            raw.keys(),
            key=lambda cid: raw[cid].get("median_f9_sub_tier", 0),
            reverse=True,  # tier cao nhất = sub nhiều nhất
        )
        for rank, cid in enumerate(sorted_ids):
            raw[cid]["name"] = TIER_NAMES[min(rank, len(TIER_NAMES) - 1)]
            raw[cid]["tier_rank"] = rank  # 0 = lớn nhất

        self._cluster_stats = raw

    def update_stats_with_labels(self, labeled_df: pd.DataFrame) -> None:
        """
        Cập nhật viral rates + per-feature percentiles sau khi có within-cluster labels.
        Gọi sau khi `fit()` và `create_labels(cluster_ids=...)` đã xong.

        Args:
            labeled_df: channel_features với cột cluster_id + is_viral_channel
        """
        if not self._is_fitted:
            raise RuntimeError("Clusterer chưa fit.")
        self._compute_cluster_stats(labeled_df)

    def _print_cluster_summary(self, df: pd.DataFrame) -> None:
        print(f"\n{'─'*60}")
        print(f"CHANNEL CLUSTERING SUMMARY (k={self.n_clusters})")
        print(f"{'─'*60}")
        for cid, stats in sorted(self._cluster_stats.items()):
            viral_rate_str = f"{stats['viral_rate']*100:.1f}%" if "viral_rate" in stats else "—"
            print(f"  Cluster {cid} | {stats['count']:3d} kênh | viral rate: {viral_rate_str}")
            print(f"    {stats['name']}")
        print(f"{'─'*60}\n")

    # ── Predict ────────────────────────────────────────────────────────────────
    def assign_cluster(
        self, features: pd.DataFrame
    ) -> tuple[int, float, dict]:
        """
        Assign cluster cho kênh mới.

        Args:
            features: DataFrame 1 dòng từ ChannelFeatureEngineer

        Returns:
            (cluster_id, distance_to_centroid, cluster_stats)
        """
        self._check_fitted()
        X = self._extract_cluster_features(features)
        X_scaled = self._scaler.transform(X)
        cluster_id = int(self._kmeans.predict(X_scaled)[0])

        # Tính khoảng cách tới centroid
        centroid = self._kmeans.cluster_centers_[cluster_id]
        distance = float(np.linalg.norm(X_scaled[0] - centroid))

        return cluster_id, distance, self._cluster_stats.get(cluster_id, {})

    # ── Save / Load ────────────────────────────────────────────────────────────
    def save(self, path: Optional[Path] = None) -> Path:
        TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = path or CLUSTERER_PATH
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Clusterer saved: %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ChannelClusterer":
        load_path = path or CLUSTERER_PATH
        with open(load_path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Clusterer loaded: %s", load_path)
        return obj

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _extract_cluster_features(self, df: pd.DataFrame) -> np.ndarray:
        available = [f for f in self.CLUSTER_FEATURES if f in df.columns]
        if not available:
            raise ValueError(f"Không có feature nào phù hợp. Cần: {self.CLUSTER_FEATURES}")
        X = df[available].fillna(0).values.copy()  # copy để tránh read-only array

        # Log transform avg_views
        avg_views_idx = available.index("f6_avg_views") if "f6_avg_views" in available else None
        if avg_views_idx is not None:
            X[:, avg_views_idx] = np.log1p(X[:, avg_views_idx])
        return X

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Clusterer chưa được fit. Gọi fit() hoặc load() trước.")

    def get_cluster_name(self, cluster_id: int) -> str:
        return self._cluster_stats.get(cluster_id, {}).get("name", f"Cluster {cluster_id}")

    def get_cluster_viral_rate(self, cluster_id: int) -> float:
        return self._cluster_stats.get(cluster_id, {}).get("viral_rate", 0.5)

    def get_within_cluster_percentile(self, cluster_id: int, feature: str, value: float) -> float:
        """
        Tính percentile rank của value so với CÁC KÊNH CÙNG CLUSTER (0–100).

        Ví dụ: MixiGaming avg_views=855K trong cluster "Established Channels"
        sẽ so sánh với các kênh 1M–10M sub khác, không phải MrBeast.

        Returns:
            Percentile rank trong cluster (0=thấp nhất, 100=cao nhất cluster).
            Trả về 50.0 nếu không có dữ liệu.
        """
        stats = self._cluster_stats.get(cluster_id, {})
        p_min = stats.get(f"min_{feature}")
        p25   = stats.get(f"p25_{feature}")
        p50   = stats.get(f"p50_{feature}")
        p75   = stats.get(f"p75_{feature}")
        p_max = stats.get(f"max_{feature}")

        if None in (p_min, p25, p50, p75, p_max):
            return 50.0  # không có dữ liệu cluster → fallback

        if value <= p_min:
            return 0.0
        if value >= p_max:
            return 100.0

        breakpoints = [
            (p_min, 0.0), (p25, 25.0), (p50, 50.0), (p75, 75.0), (p_max, 100.0),
        ]
        for i in range(len(breakpoints) - 1):
            lo_v, lo_p = breakpoints[i]
            hi_v, hi_p = breakpoints[i + 1]
            if lo_v <= value <= hi_v:
                if hi_v == lo_v:
                    return float(lo_p)
                return lo_p + (value - lo_v) / (hi_v - lo_v) * (hi_p - lo_p)
        return 50.0
