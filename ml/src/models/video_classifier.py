"""
Model B — Video Viral Prediction + Time Window Estimator.

B1: Binary Classifier (Gradient Boosting / Logistic Regression)
B2: Multi-class Time Window Classifier
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

TRAINED_MODELS_DIR = Path(__file__).parent.parent.parent / "trained_models"
MODEL_B1_PATH = TRAINED_MODELS_DIR / "model_b1_viral_classifier.pkl"
MODEL_B2_PATH = TRAINED_MODELS_DIR / "model_b2_timewindow_classifier.pkl"

VIDEO_FEATURES = [
    "v1_like_ratio",
    "v2_comment_ratio",
    "v3_like_comment_ratio",
    "v4_duration_mins",
    "v5_relative_views",
    "v6_engagement_score",
    "log_relative_views",
    "log_views",
]

EARLY_SIGNAL_FEATURES = [
    "v7_views_at_poll",
    "v8_age_hours",
    "v9_views_per_hour",
    "v10_channel_hourly_avg",
    "v11_velocity_ratio",
    "e1_views_6h",
    "e2_views_24h",
    "e3_views_48h",
    "e4_growth_rate_6_24",
    "e5_growth_rate_24_48",
]

TIME_WINDOW_CLASSES = ["not_viral", "viral_within_30d", "viral_within_7d"]

VIRAL_THRESHOLD = 0.6  # P(viral) > 0.6 → chạy B2


# ─── Sub-Model B1: Viral Binary Classifier ────────────────────────────────────
class _VideoViralB1:
    """Binary: video có viral hay không."""

    def __init__(self) -> None:
        self._model: Optional[Pipeline] = None
        self._feature_names: list[str] = []
        self._is_fitted = False

    def train(self, features_df: pd.DataFrame) -> dict:
        if "is_viral" not in features_df.columns:
            raise ValueError("Cần cột 'is_viral' để train B1.")

        available = [f for f in VIDEO_FEATURES if f in features_df.columns]
        for ef in EARLY_SIGNAL_FEATURES:
            if ef in features_df.columns and features_df[ef].notna().mean() > 0.3:
                available.append(ef)

        self._feature_names = available
        X = features_df[available].fillna(0).values
        y = features_df["is_viral"].values.astype(int)

        print(f"\n{'─'*60}")
        print("TRAINING MODEL B1 - VIDEO VIRAL CLASSIFIER")
        print(f"{'─'*60}")
        print(f"  Videos     : {len(X)}")
        print(f"  Features   : {len(available)}")
        print(f"  Viral (1)  : {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"  Not Viral  : {(1-y).sum()}")

        # Baseline: always predict majority
        majority = int(np.bincount(y).argmax())
        baseline_f1 = f1_score(y, [majority] * len(y), zero_division=0)
        print(f"  Baseline F1 (majority): {baseline_f1:.3f}")

        # Pick algorithm based on class balance
        viral_rate = y.mean()
        sample_weight = None
        if viral_rate < 0.1:
            # Imbalanced: dùng GradientBoosting + sample_weight thay vì LR balanced.
            # LR balanced inflate recall → ~97% recall nhưng precision rất thấp → false positive tràn.
            # GradientBoosting nhạy cảm hơn với threshold và ít bias hơn.
            logger.warning("Mất cân bằng (%.1f%% viral) — GradientBoosting + sample_weight", viral_rate * 100)
            algo = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, random_state=42,
            )
            sample_weight = compute_class_weight("balanced", classes=np.unique(y), y=y)
            sample_weight = sample_weight[y]  # per-sample weight
        elif len(X) >= 200:
            algo = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
        else:
            algo = RandomForestClassifier(
                n_estimators=100, max_depth=4, class_weight="balanced", random_state=42
            )

        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", algo),
        ])

        # Stratified K-Fold CV
        # Note: sample_weight chỉ dùng cho final fit — CV dùng unweighted để tránh
        # incompatibility với các phiên bản sklearn cũ hơn.
        n_splits = min(5, max(2, int(y.sum() // 5) if y.sum() >= 10 else 2))
        try:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            y_pred = cross_val_predict(self._model, X, y, cv=cv)
            cv_f1 = f1_score(y, y_pred, zero_division=0)
        except Exception as e:
            logger.warning("CV thất bại (%s) — dùng train set evaluation", e)
            self._model.fit(X, y)
            y_pred = self._model.predict(X)
            cv_f1 = f1_score(y, y_pred, zero_division=0)

        # Final fit trên toàn bộ data (có sample_weight nếu imbalanced)
        if sample_weight is not None:
            self._model.fit(X, y, model__sample_weight=sample_weight)
        else:
            self._model.fit(X, y)
        self._is_fitted = True

        print(f"\n  {'─'*40}")
        print(classification_report(y, y_pred, target_names=["Not Viral", "Viral"],
                                    zero_division=0))
        print(f"  CV F1-Score : {cv_f1:.3f}")
        print(f"  vs Baseline : {'BEAT' if cv_f1 > baseline_f1 else 'DID NOT BEAT'} "
              f"baseline ({baseline_f1:.3f})")

        return {"cv_f1": cv_f1, "baseline_f1": baseline_f1, "beat_baseline": cv_f1 > baseline_f1}

    def predict_proba(self, features: pd.DataFrame) -> float:
        self._check_fitted()
        X = self._prepare_input(features)
        try:
            proba = self._model.predict_proba(X)[0, 1]
        except Exception:
            proba = float(self._model.predict(X)[0])
        return float(proba)

    def _prepare_input(self, df: pd.DataFrame) -> np.ndarray:
        result = pd.DataFrame(index=range(len(df)))
        for feat in self._feature_names:
            result[feat] = df[feat].values if feat in df.columns else 0.0
        return result.fillna(0).values

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("B1 chưa được train.")

    def get_feature_importances(self) -> dict[str, float]:
        if not self._is_fitted:
            return {}
        m = self._model.named_steps.get("model")
        if hasattr(m, "feature_importances_"):
            return dict(zip(self._feature_names, m.feature_importances_))
        if hasattr(m, "coef_"):
            coef = np.abs(m.coef_[0])
            return dict(zip(self._feature_names, coef / (coef.sum() + 1e-8)))
        return {}


# ─── Sub-Model B2: Time Window Classifier ────────────────────────────────────
class _VideoTimeWindowB2:
    """Multi-class: "not_viral" / "viral_within_30d" / "viral_within_7d"."""

    def __init__(self) -> None:
        self._model: Optional[Pipeline] = None
        self._feature_names: list[str] = []
        self._le = LabelEncoder()
        self._is_fitted = False
        self._n_classes = 0

    def train(self, features_df: pd.DataFrame) -> dict:
        label_col = "time_window_label"
        if label_col not in features_df.columns:
            # Fallback to binary
            logger.warning("Không có time_window_label — dùng binary viral label")
            features_df = features_df.copy()
            if "is_viral" in features_df.columns:
                features_df[label_col] = features_df["is_viral"].map(
                    {1: "viral_within_30d", 0: "not_viral"}
                )
            else:
                raise ValueError("Cần time_window_label hoặc is_viral để train B2.")

        available = [f for f in VIDEO_FEATURES if f in features_df.columns]
        self._feature_names = available

        X = features_df[available].fillna(0).values
        y_raw = features_df[label_col].fillna("not_viral").values
        y = self._le.fit_transform(y_raw)
        self._n_classes = len(self._le.classes_)

        print(f"\n{'─'*60}")
        print("TRAINING MODEL B2 - TIME WINDOW CLASSIFIER")
        print(f"{'─'*60}")
        print(f"  Videos  : {len(X)}")
        print(f"  Classes : {list(self._le.classes_)}")
        for cls, lbl in zip(self._le.classes_, range(self._n_classes)):
            count = (y == lbl).sum()
            print(f"    {cls:<25}: {count:4d} ({count/len(y)*100:.1f}%)")

        # Chọn từng class — đủ minimal samples mới train multiclass
        min_count = min(np.bincount(y))
        if min_count < 3:
            logger.warning("Một số class quá ít samples (%d) — simplify thành binary", min_count)
            # Simplify
            simplify_map: dict[str, str] = {
                "viral_within_7d": "viral",
                "viral_within_30d": "viral",
                "not_viral": "not_viral",
            }
            y_bin = pd.Series(y_raw).map(simplify_map).fillna("not_viral").values
            y = self._le.fit_transform(y_bin)
            self._n_classes = len(self._le.classes_)

        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=500,
                random_state=42,
                solver="lbfgs",
            )),
        ])

        n_splits = min(5, max(2, min(np.bincount(y))))
        try:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            y_pred = cross_val_predict(self._model, X, y, cv=cv)
            cv_f1 = float(f1_score(y, y_pred, average="weighted", zero_division=0))
        except Exception as e:
            logger.warning("B2 CV thất bại (%s) — train set eval", e)
            self._model.fit(X, y)
            y_pred = self._model.predict(X)
            cv_f1 = float(f1_score(y, y_pred, average="weighted", zero_division=0))

        self._model.fit(X, y)
        self._is_fitted = True

        print(f"\n  CV F1 (weighted): {cv_f1:.3f}")
        print(classification_report(y, y_pred, target_names=self._le.classes_, zero_division=0))

        return {"cv_f1_weighted": cv_f1}

    def predict(self, features: pd.DataFrame) -> tuple[str, float]:
        """Returns (time_window_label, probability)."""
        self._check_fitted()
        X = self._prepare_input(features)
        pred_idx = int(self._model.predict(X)[0])
        try:
            proba = float(np.max(self._model.predict_proba(X)[0]))
        except Exception:
            proba = 0.6
        label = str(self._le.inverse_transform([pred_idx])[0])
        return label, proba

    def _prepare_input(self, df: pd.DataFrame) -> np.ndarray:
        result = pd.DataFrame(index=range(len(df)))
        for feat in self._feature_names:
            result[feat] = df[feat].values if feat in df.columns else 0.0
        return result.fillna(0).values

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("B2 chưa được train.")


# ─── Combined Model B ─────────────────────────────────────────────────────────
class VideoViralClassifier:
    """
    Model B = B1 (viral binary) + B2 (time window).

    Usage:
        clf = VideoViralClassifier()
        results = clf.train(video_features_df)  # df có is_viral + time_window_label
        report = clf.predict(new_video_features_df)
    """

    def __init__(self) -> None:
        self._b1 = _VideoViralB1()
        self._b2 = _VideoTimeWindowB2()
        self._is_fitted = False

    def train(self, video_features_df: pd.DataFrame) -> dict:
        """
        Train cả B1 và B2.

        Args:
            video_features_df: Kết quả từ VideoLabelCreator + VideoFeatureEngineer

        Returns:
            {'b1': {...}, 'b2': {...}}
        """
        b1_results = self._b1.train(video_features_df)
        b2_results = self._b2.train(video_features_df)
        self._is_fitted = True

        print(f"\n{'='*60}")
        print("MODEL B TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  B1 (viral binary) F1   : {b1_results['cv_f1']:.3f}")
        print(f"  B2 (time window) F1    : {b2_results['cv_f1_weighted']:.3f}")
        print(f"  Viral threshold        : {VIRAL_THRESHOLD}")
        print(f"{'='*60}\n")

        return {"b1": b1_results, "b2": b2_results}

    def predict(self, features: pd.DataFrame) -> dict:
        """
        Dự đoán viral probability + time window.

        Returns:
            {
                "will_viral": bool,
                "probability": float,
                "time_window": str,
                "time_window_probability": float,
                "label": str,
                "confidence": str,
                "has_early_signals": bool,
            }
        """
        self._check_fitted()

        # B1: viral probability
        prob = self._b1.predict_proba(features)
        will_viral = prob > VIRAL_THRESHOLD

        # B2: time window (chỉ chạy nếu B1 dự đoán viral)
        if will_viral:
            time_window, tw_prob = self._b2.predict(features)
        else:
            time_window = "not_viral"
            tw_prob = 1.0 - prob

        # Confidence
        confidence = self._calc_confidence(features, prob)

        # Label
        label = self._make_label(will_viral, prob, time_window)

        return {
            "will_viral": bool(will_viral),
            "probability": round(prob, 4),
            "time_window": time_window,
            "time_window_probability": round(tw_prob, 4),
            "label": label,
            "confidence": confidence,
            "has_early_signals": self._has_early_signals(features),
        }

    @staticmethod
    def _make_label(will_viral: bool, prob: float, time_window: str) -> str:
        if not will_viral:
            if prob > 0.4:
                return "LOW VIRAL POTENTIAL"
            return "NOT VIRAL"
        if time_window == "viral_within_7d":
            return "VIRAL WITHIN 7 DAYS"
        if time_window in ("viral_within_30d", "viral"):
            return "VIRAL WITHIN 30 DAYS"
        return "VIRAL POTENTIAL"

    @staticmethod
    def _calc_confidence(features: pd.DataFrame, prob: float) -> str:
        has_early = any(feat in features.columns and features[feat].notna().any()
                        for feat in ["e3_views_48h", "v11_velocity_ratio"])
        age_hours = float(features["v8_age_hours"].iloc[0]) if "v8_age_hours" in features.columns else 999

        if has_early and age_hours > 48:
            return "HIGH"
        if has_early or age_hours > 6:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _has_early_signals(features: pd.DataFrame) -> bool:
        return any(feat in features.columns for feat in EARLY_SIGNAL_FEATURES)

    def get_feature_importances(self) -> dict[str, float]:
        if not self._is_fitted:
            return {}
        return self._b1.get_feature_importances()

    # ── Save / Load ────────────────────────────────────────────────────────────
    def save(
        self,
        b1_path: Optional[Path] = None,
        b2_path: Optional[Path] = None,
    ) -> tuple[Path, Path]:
        TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        p1 = b1_path or MODEL_B1_PATH
        p2 = b2_path or MODEL_B2_PATH
        with open(p1, "wb") as f:
            pickle.dump(self._b1, f)
        with open(p2, "wb") as f:
            pickle.dump(self._b2, f)
        logger.info("Model B1 saved: %s", p1)
        logger.info("Model B2 saved: %s", p2)
        return p1, p2

    @classmethod
    def load(
        cls,
        b1_path: Optional[Path] = None,
        b2_path: Optional[Path] = None,
    ) -> "VideoViralClassifier":
        p1 = b1_path or MODEL_B1_PATH
        p2 = b2_path or MODEL_B2_PATH
        obj = cls()
        with open(p1, "rb") as f:
            obj._b1 = pickle.load(f)
        with open(p2, "rb") as f:
            obj._b2 = pickle.load(f)
        obj._is_fitted = True
        logger.info("Model B loaded")
        return obj

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model B chưa train. Gọi train() hoặc load().")
