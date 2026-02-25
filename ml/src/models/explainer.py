"""
Prediction Explainer — tạo giải thích ngôn ngữ tự nhiên (tiếng Việt)
cho output của Model A và Model B.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Explanation Building Blocks ──────────────────────────────────────────────
_CHANNEL_FACTOR_TEMPLATES: dict[str, dict] = {
    "f1_efficiency": {
        "high": "Kênh có hiệu suất views/subscriber vượt trội — nội dung thu hút được nhiều người xem vượt xa lượng subscriber.",
        "medium": "Hiệu suất views/subscriber ở mức trung bình so với benchmark.",
        "low": "Hiệu suất views/subscriber còn thấp — kênh chưa khai thác hết tiềm năng subscriber.",
    },
    "f2_loyalty": {
        "high": "Tỷ lệ like/view cao — khán giả phản hồi tích cực và trung thành với kênh.",
        "medium": "Tỷ lệ like/view ở mức bình thường.",
        "low": "Tỷ lệ like/view thấp — có thể khán giả chưa thực sự gắn bó với nội dung.",
    },
    "f3_depth": {
        "high": "Tỷ lệ comment/view cao — nội dung tạo ra nhiều thảo luận và tương tác sâu.",
        "medium": "Mức độ thảo luận bình thường.",
        "low": "Ít comment so với số views — nội dung chưa kích thích thảo luận.",
    },
    "f4_consistency": {
        "high": "Upload đều đặn và nhất quán — yếu tố quan trọng để duy trì và xây dựng khán giả.",
        "medium": "Tần suất upload tương đối ổn định.",
        "low": "Upload không đều — có thể làm giảm lượt subscribe và giữ chân khán giả.",
    },
    "f7_engagement": {
        "high": "Điểm engagement tổng hợp rất cao — kênh tạo ra nhiều tương tác trên mọi loại phản hồi.",
        "medium": "Điểm engagement ở mức trung bình.",
        "low": "Điểm engagement thấp — kênh cần cải thiện cách tương tác với khán giả.",
    },
    "f11_recent_trend": {
        "high": "Xu hướng tăng trưởng trong video gần nhất — kênh đang trên đà phát triển tốt.",
        "medium": "Tăng trưởng ổn định.",
        "low": "Xu hướng 5 video gần đây thấp hơn trước — có dấu hiệu chậm lại.",
    },
}

_VIDEO_FACTOR_TEMPLATES: dict[str, dict] = {
    "v1_like_ratio": {
        "high": "Tỷ lệ like/view cao bất thường — khán giả phản hồi rất tích cực ngay từ đầu.",
        "medium": "Tỷ lệ like/view ở mức bình thường.",
        "low": "Tỷ lệ like/view thấp — phản hồi chưa mạnh.",
    },
    "v2_comment_ratio": {
        "high": "Nhiều comment so với số views — video tạo ra thảo luận sôi nổi.",
        "medium": "Mức độ thảo luận bình thường.",
        "low": "Ít comment — video chưa kích thích được phản hồi từ người xem.",
    },
    "v5_relative_views": {
        "high": "Views của video này cao hơn đáng kể so với trung bình kênh — tín hiệu viral rõ ràng.",
        "medium": "Views ngang mức trung bình kênh.",
        "low": "Views thấp hơn trung bình kênh — video chưa được thuật toán đẩy mạnh.",
    },
    "v11_velocity_ratio": {
        "high": "Tốc độ tăng views/giờ vượt xa mức bình thường của kênh — nội dung đang được share mạnh.",
        "medium": "Tốc độ tăng views bình thường.",
        "low": "Tốc độ tăng views chậm — cần thêm thời gian mới có thể đánh giá chính xác.",
    },
    "e4_growth_rate_6_24": {
        "high": "Tốc độ tăng trưởng từ 6h→24h rất mạnh — video đang bùng nổ sau period đầu.",
        "medium": "Tăng trưởng bình thường trong 24h đầu.",
        "low": "Tốc độ tăng trưởng chậm lại sau 24h — momentum có thể đang giảm.",
    },
}


class PredictionExplainer:
    """
    Tạo explanation bằng tiếng Việt cho predictions của Model A và Model B.
    """

    def __init__(self) -> None:
        self._channel_fe_percentiles: dict = {}
        self._video_feature_stats: dict = {}

    def fit(
        self,
        channel_fe_percentiles: Optional[dict] = None,
        video_features_df: Optional[pd.DataFrame] = None,
    ) -> "PredictionExplainer":
        """
        Học distribution của features từ training data để so sánh.

        Args:
            channel_fe_percentiles: Từ ChannelFeatureEngineer.get_percentiles()
            video_features_df:      DataFrame chứa video features để tính stats
        """
        if channel_fe_percentiles:
            self._channel_fe_percentiles = channel_fe_percentiles

        if video_features_df is not None:
            numeric_cols = video_features_df.select_dtypes(include="number").columns
            for col in numeric_cols:
                vals = video_features_df[col].dropna()
                if len(vals) > 0:
                    self._video_feature_stats[col] = {
                        "p25": float(vals.quantile(0.25)),
                        "p50": float(vals.quantile(0.50)),
                        "p75": float(vals.quantile(0.75)),
                        "mean": float(vals.mean()),
                    }
        return self

    # ── Channel Explanation ────────────────────────────────────────────────────
    def explain_channel(
        self,
        features: pd.DataFrame,
        probability: float,
        cluster_name: str = "",
        feature_importances: Optional[dict[str, float]] = None,
    ) -> dict:
        """
        Tạo explanation cho kênh.

        Returns:
            {
                "summary": str,
                "factors": [{"feature": str, "impact": str, "description": str}],
                "risk_factors": [str],
                "strengths": [str],
            }
        """
        factors = []
        strengths = []
        risks = []

        # Xác định mức độ ưu tiên feature theo importance
        priority_features = list(_CHANNEL_FACTOR_TEMPLATES.keys())
        if feature_importances:
            priority_features = sorted(
                [f for f in feature_importances if f in _CHANNEL_FACTOR_TEMPLATES],
                key=lambda x: feature_importances.get(x, 0),
                reverse=True,
            )

        for feat_name in priority_features[:5]:  # top 5 features
            if feat_name not in features.columns:
                continue
            val = float(features[feat_name].iloc[0])
            level = self._get_level(feat_name, val, "channel")
            template = _CHANNEL_FACTOR_TEMPLATES.get(feat_name, {})
            desc = template.get(level, "")
            if not desc:
                continue

            impact = "positive" if level == "high" else ("negative" if level == "low" else "neutral")
            factors.append({
                "feature": feat_name,
                "impact": impact,
                "description": desc,
                "value": self._format_value(feat_name, val),
                "percentile": self._get_percentile(feat_name, val, "channel"),
            })
            if level == "high":
                strengths.append(desc)
            elif level == "low":
                risks.append(desc)

        summary = self._build_channel_summary(probability, cluster_name, len(strengths), len(risks))

        return {
            "summary": summary,
            "factors": factors,
            "risk_factors": risks,
            "strengths": strengths,
        }

    # ── Video Explanation ──────────────────────────────────────────────────────
    def explain_video(
        self,
        features: pd.DataFrame,
        prediction: dict,
        channel_name: str = "",
    ) -> dict:
        """
        Tạo explanation cho video.

        Returns:
            {
                "summary": str,
                "factors": [{"feature": str, "impact": str, "description": str}],
                "warnings": [str],
                "momentum_score": int,
            }
        """
        factors = []
        warnings = []

        priority_features = list(_VIDEO_FACTOR_TEMPLATES.keys())
        for feat_name in priority_features:
            if feat_name not in features.columns:
                continue
            val = float(features[feat_name].iloc[0])
            level = self._get_level(feat_name, val, "video")
            template = _VIDEO_FACTOR_TEMPLATES.get(feat_name, {})
            desc = template.get(level, "")
            if not desc:
                continue
            impact = "positive" if level == "high" else ("negative" if level == "low" else "neutral")
            factors.append({
                "feature": feat_name,
                "impact": impact,
                "description": desc,
                "value": self._format_value(feat_name, val),
            })
            if level == "low" and feat_name in ["v5_relative_views", "v11_velocity_ratio"]:
                warnings.append(desc)

        # Cảnh báo thiếu early signal
        if not prediction.get("has_early_signals"):
            warnings.append("Không có early signal data — độ chính xác có thể thấp hơn. "
                            "Hãy chờ ít nhất 24-48h sau khi video đăng để có dự đoán chính xác hơn.")

        confidence = prediction.get("confidence", "MEDIUM")
        if confidence == "LOW":
            warnings.append("Confidence thấp: video mới đăng, chưa đủ dữ liệu để dự đoán chắc chắn.")

        momentum = self._calc_momentum_score(features, prediction)
        summary = self._build_video_summary(prediction, channel_name, momentum)

        return {
            "summary": summary,
            "factors": factors,
            "warnings": warnings,
            "momentum_score": momentum,
        }

    # ── Projected Views ────────────────────────────────────────────────────────
    @staticmethod
    def project_views(
        current_views: int,
        views_per_hour: float,
        probability: float,
        time_window: str,
    ) -> dict:
        """
        Ước tính projected views theo decay model.
        Views = current + velocity * growth_factor * time_hours
        """
        decay = 0.7 if time_window == "viral_within_7d" else 0.4
        boost = 1.0 + probability

        # 7 ngày
        views_7d = int(current_views + views_per_hour * 24 * 7 * boost * decay)
        # 30 ngày (velocity giảm dần)
        views_30d = int(views_7d + views_per_hour * 24 * 23 * boost * (decay * 0.3))

        confidence = "HIGH" if probability > 0.75 else ("MEDIUM" if probability > 0.5 else "LOW")

        def fmt(n: int) -> str:
            if n >= 1_000_000:
                return f"~{n/1_000_000:.1f}M"
            if n >= 1_000:
                return f"~{n/1_000:.0f}K"
            return f"~{n}"

        return {
            "7_days": fmt(views_7d),
            "30_days": fmt(views_30d),
            "confidence": confidence,
        }

    # ── Private Helpers ────────────────────────────────────────────────────────
    def _get_level(self, feat: str, val: float, domain: str) -> str:
        """Phân loại val thành high / medium / low dựa trên distribution."""
        stats_dict = (
            self._channel_fe_percentiles if domain == "channel"
            else self._video_feature_stats
        )
        if feat not in stats_dict:
            return "medium"
        stats = stats_dict[feat]
        if val >= stats.get("p75", stats.get("mean", 0) * 1.5):
            return "high"
        if val <= stats.get("p25", stats.get("mean", 0) * 0.5):
            return "low"
        return "medium"

    def _get_percentile(self, feat: str, val: float, domain: str) -> Optional[int]:
        stats_dict = (
            self._channel_fe_percentiles if domain == "channel"
            else self._video_feature_stats
        )
        if feat not in stats_dict:
            return None
        stats = stats_dict[feat]
        lo, hi = stats.get("min", 0), stats.get("max", 1)
        if hi == lo:
            return 50
        return max(0, min(100, int((val - lo) / (hi - lo) * 100)))

    @staticmethod
    def _format_value(feat: str, val: float) -> str:
        if "ratio" in feat or "loyalty" in feat or "depth" in feat:
            return f"{val*100:.2f}%"
        if "views" in feat.lower() and val > 1000:
            if val >= 1_000_000:
                return f"{val/1_000_000:.1f}M"
            return f"{val/1_000:.0f}K"
        if "tier" in feat:
            return f"{10**val:,.0f} subscribers"
        if "trend" in feat:
            return f"{val:.2f}x" if val != 0 else "N/A"
        return f"{val:.3f}"

    @staticmethod
    def _build_channel_summary(
        prob: float, cluster_name: str, n_strengths: int, n_risks: int
    ) -> str:
        pct = int(prob * 100)
        tier = (
            "cực cao" if prob > 0.85
            else "cao" if prob > 0.70
            else "trung bình" if prob > 0.50
            else "thấp"
        )
        base = f"Kênh này có xác suất {pct}% tạo ra video viral — tiềm năng ở mức {tier}."
        if cluster_name:
            base += f" Kênh thuộc nhóm '{cluster_name}'."
        if n_strengths > 0:
            base += f" Có {n_strengths} điểm mạnh nổi bật."
        if n_risks > 0:
            base += f" Lưu ý {n_risks} yếu tố cần cải thiện."
        return base

    @staticmethod
    def _build_video_summary(prediction: dict, channel_name: str, momentum: int) -> str:
        prob = int(prediction.get("probability", 0) * 100)
        will_viral = prediction.get("will_viral", False)
        time_window = prediction.get("time_window", "")
        confidence = prediction.get("confidence", "MEDIUM")

        if will_viral:
            tw_map = {
                "viral_within_7d": "trong vòng 7 ngày",
                "viral_within_30d": "trong vòng 30 ngày",
                "viral": "trong tháng tới",
            }
            tw_str = tw_map.get(time_window, "trong thời gian tới")
            base = (
                f"Video có xác suất viral {prob}% — dự kiến viral {tw_str}. "
                f"Momentum score: {momentum}/100."
            )
        else:
            base = (
                f"Video chưa có dấu hiệu viral rõ ràng (xác suất {prob}%). "
                f"Có thể cần thêm thời gian để đánh giá."
            )

        if confidence == "LOW":
            base += " (Độ tin cậy thấp — cần thêm data)"
        elif confidence == "MEDIUM":
            base += " (Độ tin cậy trung bình)"

        return base

    @staticmethod
    def _calc_momentum_score(features: pd.DataFrame, prediction: dict) -> int:
        """Score 0-100 đại diện cho momentum hiện tại của video."""
        score = 50  # base

        prob = prediction.get("probability", 0.5)
        score += int((prob - 0.5) * 60)

        if "v11_velocity_ratio" in features.columns:
            vr = float(features["v11_velocity_ratio"].iloc[0])
            if vr > 3:
                score += 20
            elif vr > 1.5:
                score += 10
            elif vr < 0.5:
                score -= 15

        if "e4_growth_rate_6_24" in features.columns:
            gr = float(features["e4_growth_rate_6_24"].iloc[0])
            if gr > 2:
                score += 15
            elif gr > 0.5:
                score += 5

        return max(0, min(100, score))
