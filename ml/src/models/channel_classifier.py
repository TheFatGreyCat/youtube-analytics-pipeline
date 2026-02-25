"""
Model A — Channel Viral Potential Classifier.

Algorithm: Random Forest (primary) với fallback Logistic Regression / SVM.
Training: Leave-One-Out Cross Validation (LOOCV) vì chỉ có 42 samples.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

TRAINED_MODELS_DIR = Path(__file__).parent.parent.parent / "trained_models"
MODEL_A_PATH = TRAINED_MODELS_DIR / "model_a_channel_classifier.pkl"

CHANNEL_FEATURES = [
    "f1_efficiency",
    "f2_loyalty",
    "f3_depth",
    "f4_consistency",
    "f6_avg_views",
    "f7_engagement",
    "f9_sub_tier",
    "f11_recent_trend",
    # cluster features sẽ thêm sau khi clustering
]

# Optional cluster features — chỉ giữ cluster_distance (importance cao), bỏ cluster_id (importance=0)
CLUSTER_FEATURES = ["cluster_distance"]


class ChannelViralClassifier:
    """
    Model A: dự đoán kênh có viral potential hay không.

    Usage:
        clf = ChannelViralClassifier()
        report = clf.train(channel_features_df)  # df có cột is_viral_channel
        prob = clf.predict_proba(new_features_df)
        ChannelViralClassifier.load()
    """

    def __init__(self) -> None:
        self._model: Optional[Pipeline] = None
        self._feature_names: list[str] = []
        self._loocv_results: dict = {}
        self._is_fitted = False

    # ── Training ───────────────────────────────────────────────────────────────
    def train(self, labeled_features: pd.DataFrame) -> dict:
        """
        Train Model A với LOOCV evaluation.

        Args:
            labeled_features: DataFrame với cột feature + is_viral_channel

        Returns:
            dict chứa LOOCV metrics
        """
        if "is_viral_channel" not in labeled_features.columns:
            raise ValueError("Cần cột 'is_viral_channel' để train.")

        # Chọn features có trong data
        available_feats = [f for f in CHANNEL_FEATURES if f in labeled_features.columns]
        for cf in CLUSTER_FEATURES:
            if cf in labeled_features.columns:
                available_feats.append(cf)

        self._feature_names = available_feats
        X = labeled_features[available_feats].fillna(0).values
        y = labeled_features["is_viral_channel"].values.astype(int)

        print(f"\n{'='*60}")
        print(f"🤖 TRAINING MODEL A — CHANNEL VIRAL CLASSIFIER")
        print(f"{'='*60}")
        print(f"  Số kênh: {len(X)}")
        print(f"  Features: {available_feats}")
        print(f"  Label 1 (viral): {y.sum()} | Label 0: {(1-y).sum()}")
        print(f"  Training strategy: Leave-One-Out CV (LOOCV)")

        # ── Thử 3 algorithms ── ───────────────────────────────────────────
        candidates = {
            "random_forest": Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=2,       # giảm từ 3 → 2 để tránh overfit
                    min_samples_leaf=5, # tăng từ 3 → 5
                    class_weight="balanced",
                    random_state=42,
                )),
            ]),
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    class_weight="balanced",
                    C=0.5,             # regularization mạnh hơn
                    max_iter=500,
                    random_state=42,
                )),
            ]),
            "svm": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                )),
            ]),
        }

        best_name = ""
        best_f1 = -1.0
        all_results: dict[str, dict] = {}

        print(f"\n{'─'*60}")
        print(f"  LOOCV RESULTS:")
        print(f"{'─'*60}")

        for name, pipeline in candidates.items():
            results = self._run_loocv(pipeline, X, y)
            all_results[name] = results
            f1 = results["f1"]
            acc = results["accuracy"]
            print(f"  {name:<25} | acc={acc:.3f} | f1={f1:.3f} | "
                  f"prec={results['precision']:.3f} | rec={results['recall']:.3f}")
            if f1 > best_f1:
                best_f1 = f1
                best_name = name

        # ── Overfit guard: nếu F1=1.0 (42 mẫu), ưu tiên model ít overfit hơn ──
        if best_f1 >= 1.0:
            print(f"\n  ⚠️  F1=1.0 trên {len(X)} mẫu → có thể overfit.")
            # Ưu tiên logistic_regression nếu F1 >= 0.65
            lr_f1 = all_results.get("logistic_regression", {}).get("f1", 0)
            if lr_f1 >= 0.65:
                best_name = "logistic_regression"
                best_f1 = lr_f1
                print(f"  → Chọn logistic_regression (F1={lr_f1:.3f}) để tổng quát hơn.")
            else:
                print(f"  → Giữ {best_name} nhưng lưu ý kết quả có thể lạc quan.")

        print(f"\n  ✅ Best model: {best_name} (F1={best_f1:.3f})")

        # ── Train final model trên full data ───────────────────────────────
        self._model = candidates[best_name]
        self._model.fit(X, y)
        self._is_fitted = True

        # Feature importance chỉ có với RF
        if best_name == "random_forest":
            self._print_feature_importance()

        self._loocv_results = {
            "best_model": best_name,
            "all_results": all_results,
            **all_results[best_name],
        }

        print(f"\n{'='*60}")
        print(f"📊 FINAL MODEL A METRICS (LOOCV)")
        print(f"{'='*60}")
        print(f"  Model    : {best_name}")
        print(f"  Accuracy : {self._loocv_results['accuracy']:.3f}")
        print(f"  F1-Score : {self._loocv_results['f1']:.3f}")
        print(f"  Precision: {self._loocv_results['precision']:.3f}")
        print(f"  Recall   : {self._loocv_results['recall']:.3f}")
        print(f"{'='*60}\n")

        return self._loocv_results

    @staticmethod
    def _run_loocv(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
        loo = LeaveOneOut()
        preds = np.zeros(len(y), dtype=int)
        probs = np.zeros(len(y))

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            pipeline.fit(X_train, y_train)
            preds[test_idx] = pipeline.predict(X_test)
            try:
                probs[test_idx] = pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                probs[test_idx] = float(preds[test_idx])

        return {
            "accuracy": accuracy_score(y, preds),
            "f1": f1_score(y, preds, zero_division=0),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
        }

    def _print_feature_importance(self) -> None:
        if not self._model:
            return
        rf = self._model.named_steps.get("model")
        if not hasattr(rf, "feature_importances_"):
            return

        importances = rf.feature_importances_
        feature_imp = sorted(
            zip(self._feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"\n  📊 Feature Importances (Random Forest):")
        print(f"  {'Feature':<25} {'Importance':>10}")
        print(f"  {'─'*25} {'─'*10}")
        for feat, imp in feature_imp:
            bar = "█" * int(imp * 50)
            print(f"  {feat:<25} {imp:>.3f}  {bar}")

    # ── Predict ────────────────────────────────────────────────────────────────
    def predict_proba(self, features: pd.DataFrame) -> tuple[float, str]:
        """
        Dự đoán xác suất viral cho kênh.

        Returns:
            (probability_float, confidence_label)
        """
        self._check_fitted()
        X = self._prepare_input(features)
        prob = float(self._model.predict_proba(X)[0, 1])
        confidence = self._calc_confidence(features, prob)
        return prob, confidence

    def get_feature_importances(self) -> dict[str, float]:
        """Trả về dict {feature_name: importance}."""
        self._check_fitted()
        rf = self._model.named_steps.get("model")
        if hasattr(rf, "feature_importances_"):
            return dict(zip(self._feature_names, rf.feature_importances_))
        # Logistic Regression → dùng abs(coef)
        if hasattr(rf, "coef_"):
            coef = np.abs(rf.coef_[0])
            coef_norm = coef / (coef.sum() + 1e-8)
            return dict(zip(self._feature_names, coef_norm))
        return {}

    # ── Save / Load ────────────────────────────────────────────────────────────
    def save(self, path: Optional[Path] = None) -> Path:
        TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = path or MODEL_A_PATH
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        logger.info("✅ Model A đã lưu: %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ChannelViralClassifier":
        load_path = path or MODEL_A_PATH
        with open(load_path, "rb") as f:
            obj = pickle.load(f)
        logger.info("✅ Model A đã tải: %s", load_path)
        return obj

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _prepare_input(self, features: pd.DataFrame) -> np.ndarray:
        available = [f for f in self._feature_names if f in features.columns]
        missing = [f for f in self._feature_names if f not in features.columns]
        if missing:
            logger.warning("Kênh mới thiếu features: %s — dùng giá trị 0", missing)

        X = pd.DataFrame(index=range(len(features)))
        for feat in self._feature_names:
            if feat in features.columns:
                X[feat] = features[feat].values
            else:
                X[feat] = 0.0
        return X.fillna(0).values

    @staticmethod
    def _calc_confidence(features: pd.DataFrame, prob: float) -> str:
        """
        Confidence dựa trên data source và probability certainty.
        - HIGH: BigQuery data + prob rõ ràng (>0.75 hoặc <0.25)
        - MEDIUM: API data hoặc probability không rõ (0.4-0.6)
        - LOW: thiếu nhiều features
        """
        from_api = "f11_recent_trend" in features.columns and features["f11_recent_trend"].notna().any()
        prob_certain = prob > 0.75 or prob < 0.25

        if prob_certain and not from_api:
            return "HIGH"
        if prob_certain or from_api:
            return "MEDIUM"
        return "LOW"

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model A chưa được train. Gọi train() hoặc load() trước.")

    @property
    def loocv_results(self) -> dict:
        return self._loocv_results
