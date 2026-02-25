"""
Viral Prediction System — Main Pipeline.
Tổng hợp tất cả components: BigQuery, API, Feature Engineering, Models, Explainer.
"""
from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ml.src.pipeline.report_generator import ChannelReport, VideoReport

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TRAINED_MODELS_DIR = Path(__file__).parent.parent.parent / "trained_models"
SYSTEM_STATE_PATH = TRAINED_MODELS_DIR / "system_state.pkl"


class ViralPredictionSystem:
    """
    Main entry point cho toàn bộ hệ thống dự đoán viral.

    Modes:
    1. Training mode: train(load_from_bigquery=True)
    2. Inference mode: predict_channel("MrBeast") | predict_video("MrBeast")

    Usage (training):
        system = ViralPredictionSystem()
        system.train()
        system.save()

    Usage (inference):
        system = ViralPredictionSystem.load()
        report = system.predict_channel("MrBeast")
        print(report.to_dict())
    """

    def __init__(self) -> None:
        # Lazy imports để tránh circular dependency
        from ml.src.data.bigquery_loader import BigQueryLoader
        from ml.src.data.feature_engineer import (
            ChannelFeatureEngineer,
            VideoFeatureEngineer,
        )
        from ml.src.models.channel_classifier import ChannelViralClassifier
        from ml.src.models.channel_clusterer import ChannelClusterer
        from ml.src.models.explainer import PredictionExplainer
        from ml.src.models.label_creator import ChannelLabelCreator, VideoLabelCreator
        from ml.src.models.video_classifier import VideoViralClassifier
        from ml.src.pipeline.report_generator import ChannelReport, VideoReport

        self._bq_loader = BigQueryLoader()
        self._channel_fe = ChannelFeatureEngineer()
        self._video_fe = VideoFeatureEngineer()
        self._channel_label_creator = ChannelLabelCreator()
        self._video_label_creator = VideoLabelCreator()
        self._clusterer = ChannelClusterer()
        self._model_a = ChannelViralClassifier()
        self._model_b = VideoViralClassifier()
        self._explainer = PredictionExplainer()
        self._is_trained = False

        # Kết nối API Client (lazy — chỉ khởi tạo khi cần)
        self._api_client = None
        self._ChannelReport = ChannelReport
        self._VideoReport = VideoReport

    def _get_api_client(self):
        if self._api_client is None:
            from ml.src.data.youtube_client import YouTubeAPIClient
            self._api_client = YouTubeAPIClient()
        return self._api_client

    # ── Training ───────────────────────────────────────────────────────────────
    def train(self, auto_find_k: bool = True) -> dict:
        """
        End-to-end training pipeline.
        Bước 0: Load data + Create labels
        Bước 1: Feature engineering
        Bước 2: Clustering
        Bước 3: Train Model A (LOOCV)
        Bước 4: Train Model B (stratified CV)

        Returns:
            dict với training metrics
        """
        print("\n" + "="*70)
        print("STARTING END-TO-END TRAINING PIPELINE")
        print("="*70)

        # ── Bước 0: Load data ─────────────────────────────────────────────
        print("\nStep 0: Loading BigQuery data...")
        channel_df, engagement_df, video_df = self._bq_loader.load_all()

        if channel_df.empty:
            raise RuntimeError("Không tải được int_channel_summary — kiểm tra BigQuery connection.")

        # ── Bước 0b: Video labels (không phụ thuộc clustering) ───────────
        print("\nStep 0b: Creating labels...")
        labeled_video = self._video_label_creator.create_labels(engagement_df, video_df)

        # ── Bước 1: Feature engineering kênh (không cần labels) ──────────
        print("\nStep 1: Feature engineering...")
        self._channel_fe.fit(channel_df)
        channel_features = self._channel_fe.transform(channel_df, video_df)

        # ── Bước 2a: Cluster để xác định tier → dùng làm within-cluster threshold ──
        print("\nStep 2: Channel clustering...")
        self._clusterer.fit(
            channel_features,
            labeled_df=None,          # chưa có labels, tính viral_rate sau
            auto_find_k=auto_find_k,
        )

        # Lấy cluster id tạm cho từng kênh
        prelim_results = [
            self._clusterer.assign_cluster(channel_features.iloc[[i]])
            for i in range(len(channel_features))
        ]
        prelim_cluster_ids = [r[0] for r in prelim_results]

        # ── Bước 0b': Channel labels (within-cluster) ─────────────────────
        labeled_channel = self._channel_label_creator.create_labels(
            channel_df, cluster_ids=prelim_cluster_ids
        )

        # Gắn labels vào channel_features
        channel_features["is_viral_channel"] = labeled_channel["is_viral_channel"].values

        # ── Bước 2b: Cập nhật cluster stats với viral rates + per-feat percentiles ─
        # Gán cluster_id + cluster_distance vào channel_features
        cluster_results = [
            self._clusterer.assign_cluster(channel_features.iloc[[i]])
            for i in range(len(channel_features))
        ]
        channel_features["cluster_id"]       = [r[0] for r in cluster_results]
        channel_features["cluster_distance"] = [r[1] for r in cluster_results]

        # update_stats_with_labels recomputes cluster stats (viral rates + percentiles)
        self._clusterer.update_stats_with_labels(channel_features)
        self._clusterer._print_cluster_summary(channel_features)

        # ── Video features + labels ───────────────────────────────────────
        self._video_fe.fit(engagement_df, video_df)
        video_features = self._video_fe.transform(engagement_df, video_df)
        for label_col in ["is_viral", "time_window_label"]:
            if label_col in labeled_video.columns:
                if "video_id" in video_features.columns and "video_id" in labeled_video.columns:
                    label_series = labeled_video.set_index("video_id")[label_col]
                    video_features[label_col] = video_features["video_id"].map(label_series)
                else:
                    video_features[label_col] = labeled_video[label_col].values[:len(video_features)]

        # ── Bước 3: Train Model A ─────────────────────────────────────────
        print("\nStep 3: Training Model A (Channel Classifier)...")
        model_a_results = self._model_a.train(channel_features)

        # ── Bước 4: Train Model B ─────────────────────────────────────────
        print("\nStep 4: Training Model B (Video Classifier)...")
        model_b_results = self._model_b.train(video_features)

        # ── Fit explainer ─────────────────────────────────────────────────
        self._explainer.fit(
            channel_fe_percentiles=self._channel_fe.get_percentiles(),
            video_features_df=video_features,
        )

        self._is_trained = True

        results = {
            "model_a": model_a_results,
            "model_b": model_b_results,
            "training_timestamp": datetime.now().isoformat(),
        }

        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print(f"   Model A (LOOCV): accuracy={model_a_results.get('accuracy', 0):.3f}, "
              f"F1={model_a_results.get('f1', 0):.3f}")
        print(f"   Model B1 (CV F1): {model_b_results.get('b1', {}).get('cv_f1', 0):.3f}")
        print(f"   Model B2 (CV F1): {model_b_results.get('b2', {}).get('cv_f1_weighted', 0):.3f}")
        print("="*70 + "\n")

        return results

    # ── Channel Prediction ─────────────────────────────────────────────────────
    def predict_channel(self, channel_name: str) -> "ChannelReport":
        """
        Dự đoán viral potential cho một kênh bất kỳ.

        Args:
            channel_name: Tên kênh (ví dụ: "MrBeast", "Bờm Vlogs")

        Returns:
            ChannelReport object với đầy đủ thông tin.
        """
        self._check_trained()
        api = self._get_api_client()

        print(f"\nAnalyzing channel: {channel_name}")

        # ── Fetch từ YouTube API ───────────────────────────────────────────
        data = api.get_channel_data_full(channel_name)
        channel_stats = data["channel"]
        video_stats = data["video_stats"]

        # ── Feature engineering ───────────────────────────────────────────
        features = self._channel_fe.transform_from_api(channel_stats, video_stats)

        # ── Cluster assignment ────────────────────────────────────────────
        cluster_id, cluster_dist, cluster_info = self._clusterer.assign_cluster(features)
        features["cluster_distance"] = cluster_dist  # cluster_id bỏ — importance=0

        # ── Predict ───────────────────────────────────────────────────────
        probability, confidence = self._model_a.predict_proba(features)

        # Blend với cluster prior
        cluster_prior = cluster_info.get("viral_rate", 0.5)
        blended_prob = 0.7 * probability + 0.3 * cluster_prior

        # ── Feature importances ───────────────────────────────────────────
        feature_importances = self._model_a.get_feature_importances()

        # Subscriber-based tier (chính xác hơn KMeans label với k nhỏ)
        subs = channel_stats.get("subscribers", 0) or 0
        if subs >= 100_000_000:
            tier_name = "Mega Channels (100M+ sub)"
        elif subs >= 10_000_000:
            tier_name = "Large Channels (10M–100M sub)"
        elif subs >= 1_000_000:
            tier_name = "Established Channels (1M–10M sub)"
        elif subs >= 100_000:
            tier_name = "Growing Channels (100K–1M sub)"
        else:
            tier_name = "Niche Channels (<100K sub)"

        # ── Explanation ───────────────────────────────────────────────────
        explanation = self._explainer.explain_channel(
            features,
            blended_prob,
            cluster_name=tier_name,
            feature_importances=feature_importances,
        )

        # ── Compute key metrics ───────────────────────────────────────────
        views_list = [v["views"] for v in video_stats.values() if v.get("views", 0) > 0]
        avg_views = int(sum(views_list) / len(views_list)) if views_list else 0
        recent_trend = float(features["f11_recent_trend"].iloc[0]) if "f11_recent_trend" in features.columns else 1.0

        # ── Build report ──────────────────────────────────────────────────
        # Benchmark % dùng within-cluster percentile:
        within_cluster_pct = self._clusterer.get_within_cluster_percentile(
            cluster_id, "f6_avg_views", avg_views
        )
        # Fallback sang global nếu cluster chưa có dữ liệu
        if within_cluster_pct == 50.0:
            within_cluster_pct = self._channel_fe.get_percentile_rank("f6_avg_views", avg_views)

        return self._ChannelReport(
            input_name=channel_name,
            channel_id=channel_stats["channel_id"],
            channel_name=channel_stats["channel_name"],
            subscribers=channel_stats["subscribers"],
            probability=round(blended_prob, 4),
            confidence=confidence,
            cluster_id=cluster_id,
            cluster_name=tier_name,
            explanation=explanation,
            avg_views_per_video=avg_views,
            like_ratio=float(features.get("f2_loyalty", pd.Series([0])).iloc[0]),
            comment_ratio=float(features.get("f3_depth", pd.Series([0])).iloc[0]),
            recent_trend=recent_trend,
            percentile_vs_benchmark=within_cluster_pct,
        )

    # ── Video Prediction ───────────────────────────────────────────────────────
    def predict_video(
        self,
        channel_name: str,
        video_id: Optional[str] = None,
    ) -> "VideoReport":
        """
        Dự đoán viral probability cho video mới nhất của kênh (hoặc video cụ thể).

        Args:
            channel_name: Tên kênh
            video_id:     Nếu None → lấy video mới nhất

        Returns:
            VideoReport object
        """
        self._check_trained()
        api = self._get_api_client()

        print(f"\nAnalyzing video - channel: {channel_name}")

        # ── Fetch channel info ────────────────────────────────────────────
        channel_info = api.search_channel(channel_name)
        channel_stats = api.get_channel_stats(channel_info["channel_id"])

        # ── Fetch video ───────────────────────────────────────────────────
        if video_id is None:
            videos = api.get_recent_videos(channel_info["channel_id"], n=1)
            if not videos:
                raise ValueError(f"Không tìm thấy video nào của kênh '{channel_name}'")
            video_id = videos[0]["video_id"]
            print(f"  Latest video: {video_id}")

        video_stats_dict = api.get_video_stats([video_id])
        if video_id not in video_stats_dict:
            raise ValueError(f"Không lấy được thông tin video: {video_id}")

        video_data = video_stats_dict[video_id]

        # ── Feature engineering ───────────────────────────────────────────
        features = self._video_fe.transform_from_api(video_data, channel_stats)

        # ── Predict ───────────────────────────────────────────────────────
        prediction = self._model_b.predict(features)

        # ── Absolute velocity boost ───────────────────────────────────────
        # Model B học từ relative views (so với avg kênh), nên kênh lớn như
        # MrBeast bị underestimate. Bổ sung tín hiệu tuyệt đối:
        views_per_hour = float(features["v9_views_per_hour"].iloc[0]) if "v9_views_per_hour" in features.columns else 0
        abs_views = video_data.get("views", 0)
        prediction = self._apply_absolute_boost(prediction, views_per_hour, abs_views)

        # ── Explanation ───────────────────────────────────────────────────
        explanation = self._explainer.explain_video(features, prediction, channel_name)

        # ── Projected views ───────────────────────────────────────────────
        # views_per_hour đã tính ở absolute boost trên
        projected = self._explainer.project_views(
            current_views=video_data.get("views", 0),
            views_per_hour=views_per_hour,
            probability=prediction["probability"],
            time_window=prediction["time_window"],
        )

        # ── Compute age ───────────────────────────────────────────────────
        published_at = video_data.get("published_at", "")
        age_str = self._format_age(published_at)

        channel_avg = self._video_fe.get_channel_avg(channel_stats["channel_id"])
        if channel_avg == 0:
            channel_avg = channel_stats.get("total_views", 1) / max(channel_stats.get("video_count", 1), 1)

        # So sánh dựa trên velocity (views/giờ) thay vì raw views,
        # tránh sai lệch với video còn non (2-3 ngày tuổi vs avg lifetime 90 ngày)
        channel_hourly_avg = channel_avg / (30 * 24)  # views/giờ kỳ vọng
        velocity_ratio = views_per_hour / max(channel_hourly_avg, 1)
        vs_avg_pct = (velocity_ratio - 1) * 100  # +X% = nhanh hơn X% so với pace trung bình

        # Percentile từ velocity: ratio=1 → top 50%, ratio=2 → top 75%
        channel_percentile = int(min(99, max(1, 50 * min(velocity_ratio, 2))))

        return self._VideoReport(
            video_id=video_id,
            video_title=video_data.get("title", ""),
            channel_name=channel_name,
            published_at=published_at,
            video_age=age_str,
            prediction=prediction,
            current_views=video_data.get("views", 0),
            views_per_hour=views_per_hour,
            vs_channel_avg_pct=vs_avg_pct,
            channel_percentile=channel_percentile,
            explanation=explanation,
            projected_views=projected,
        )

    # ── Save / Load ────────────────────────────────────────────────────────────
    def save(self) -> None:
        """Lưu tất cả components đã train."""
        TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._clusterer.save()
        self._model_a.save()
        self._model_b.save()

        # Lưu feature engineers và explainer
        with open(TRAINED_MODELS_DIR / "channel_fe.pkl", "wb") as f:
            pickle.dump(self._channel_fe, f)
        with open(TRAINED_MODELS_DIR / "video_fe.pkl", "wb") as f:
            pickle.dump(self._video_fe, f)
        with open(TRAINED_MODELS_DIR / "explainer.pkl", "wb") as f:
            pickle.dump(self._explainer, f)

        print(f"\nModels saved to: {TRAINED_MODELS_DIR}")

    @classmethod
    def load(cls) -> "ViralPredictionSystem":
        """Load system đã train từ file."""
        system = cls.__new__(cls)

        from ml.src.models.channel_classifier import ChannelViralClassifier
        from ml.src.models.channel_clusterer import ChannelClusterer
        from ml.src.models.video_classifier import VideoViralClassifier
        from ml.src.pipeline.report_generator import ChannelReport, VideoReport

        system._clusterer = ChannelClusterer.load()
        system._model_a = ChannelViralClassifier.load()
        system._model_b = VideoViralClassifier.load()

        with open(TRAINED_MODELS_DIR / "channel_fe.pkl", "rb") as f:
            system._channel_fe = pickle.load(f)
        with open(TRAINED_MODELS_DIR / "video_fe.pkl", "rb") as f:
            system._video_fe = pickle.load(f)
        with open(TRAINED_MODELS_DIR / "explainer.pkl", "rb") as f:
            system._explainer = pickle.load(f)

        system._api_client = None
        system._is_trained = True
        system._ChannelReport = ChannelReport
        system._VideoReport = VideoReport

        print("ViralPredictionSystem loaded from trained_models/")
        return system

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "System chưa được train. Gọi system.train() hoặc ViralPredictionSystem.load()."
            )

    @staticmethod
    def _apply_absolute_boost(
        prediction: dict,
        views_per_hour: float,
        abs_views: int,
    ) -> dict:
        """
        Bổ sung tín hiệu tuyệt đối vào kết quả dự đoán.

        Model B học từ relative_views nên underestimate kênh lớn có avg cao.
        Nếu views/giờ rất cao (>10K) → đây là tín hiệu viral thực sự.

        Ngưỡng tham khảo thực tế:
          >100K views/giờ  → cực kỳ viral (chỉ vài chục video/ngày toàn YouTube)
          >10K  views/giờ  → viral mạnh
          >1K   views/giờ  → tiềm năng viral
        """
        prob = prediction.get("probability", 0.0)
        boosted = prob

        if views_per_hour > 100_000:
            boosted = max(prob, 0.82)
            boost_label = "absolute_mega (>100K/h)"
        elif views_per_hour > 30_000:
            boosted = max(prob, 0.65)
            boost_label = "absolute_high (>30K/h)"
        elif views_per_hour > 10_000:
            boosted = max(prob, 0.50)
            boost_label = "absolute_medium (>10K/h)"
        elif views_per_hour > 2_000:
            boosted = max(prob, 0.35)
            boost_label = "absolute_low (>2K/h)"
        else:
            boost_label = None

        # Cũng boost nếu tổng views tuyệt đối rất cao trong ít giờ
        if abs_views > 5_000_000:
            boosted = max(boosted, 0.60)

        if boosted != prob:
            logger.info(
                "Absolute boost: %.1f%% → %.1f%% (%s)",
                prob * 100, boosted * 100, boost_label,
            )
            prediction = dict(prediction)
            prediction["probability"] = round(boosted, 4)
            # Cập nhật label nếu vượt ngưỡng
            if boosted >= 0.6 and prediction.get("will_viral") is False:
                prediction["will_viral"] = True
                prediction["label"] = "VIRAL POTENTIAL"
            # Cập nhật time_window nếu còn not_viral
            if prediction.get("time_window") == "not_viral" and boosted >= 0.65:
                prediction["time_window"] = "viral_within_30d"

        return prediction

    @staticmethod
    def _format_age(published_at: str) -> str:
        if not published_at:
            return "Không rõ"
        try:
            from datetime import timezone
            pub = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta = now - pub
            days = delta.days
            hours = delta.seconds // 3600
            if days > 0:
                return f"{days} ngày {hours} giờ"
            return f"{hours} giờ"
        except Exception:
            return "Không rõ"


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")

    # Cần chạy từ thư mục gốc project để import đúng
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("=" * 48)
    print("   HE THONG DU DOAN VIRAL YOUTUBE")
    print("=" * 48)

    print("Dang load models...")
    system = ViralPredictionSystem.load()
    print("Load xong!\n")

    while True:
        try:
            channel = input(">>> Ten kenh (exit de thoat): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTam biet!")
            break

        if not channel or channel.lower() == "exit":
            print("Tam biet!")
            break

        print()
        try:
            print("--- Phan tich kenh ---")
            cr = system.predict_channel(channel)
            cr.print_report()

            print()
            print("--- Video moi nhat ---")
            vr = system.predict_video(channel)
            vr.print_report()
        except Exception as e:
            print(f"Loi: {e}")

        print("\n" + "-" * 48 + "\n")
