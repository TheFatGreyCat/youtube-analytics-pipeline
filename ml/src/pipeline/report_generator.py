"""
Report Generator — chuẩn hoá output thành ChannelReport và VideoReport.
Format tuân theo spec trong prompt (Bước 7).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


def _fmt_number(n: int | float) -> str:
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


def _viral_label(prob: float) -> str:
    if prob >= 0.85:
        return "EXTREMELY HIGH"
    if prob >= 0.70:
        return "HIGH"
    if prob >= 0.55:
        return "MEDIUM"
    if prob >= 0.40:
        return "LOW"
    return "VERY LOW"


# ─── Channel Report ────────────────────────────────────────────────────────────
@dataclass
class ChannelReport:
    input_name: str
    channel_id: str
    channel_name: str
    subscribers: int
    probability: float
    confidence: str
    cluster_id: int
    cluster_name: str
    explanation: dict
    avg_views_per_video: int = 0
    like_ratio: float = 0.0
    comment_ratio: float = 0.0
    recent_trend: float = 1.0
    percentile_vs_benchmark: float = 50.0

    def to_dict(self) -> dict:
        upload_freq = "N/A"
        trend_emoji = "Tang truong" if self.recent_trend > 1.05 else (
            "Giam" if self.recent_trend < 0.95 else "On dinh"
        )
        # Clip to [5, 95] so displayed rank never shows "top 0%" or "top 100%"
        p = max(5, min(95, int(self.percentile_vs_benchmark)))

        return {
            "input": self.input_name,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "subscribers": self.subscribers,

            "viral_potential": {
                "probability": self.probability,
                "label": _viral_label(self.probability),
                "confidence": self.confidence,
                "percentile_vs_benchmark": p,
            },

            "cluster": {
                "id": self.cluster_id,
                "name": self.cluster_name,
                "description": self._cluster_desc(),
            },

            "key_metrics": {
                "avg_views_per_video": _fmt_number(self.avg_views_per_video),
                "like_ratio": f"{self.like_ratio*100:.2f}% (top {max(1, 100-p)}% trong tier)",
                "comment_ratio": f"{self.comment_ratio*100:.2f}%",
                "upload_frequency": upload_freq,
            },

            "explanation": self.explanation.get("factors", []),
            "summary": self.explanation.get("summary", ""),
            "recent_trend": trend_emoji,
            "risk_factors": self.explanation.get("risk_factors", []),
            "strengths": self.explanation.get("strengths", []),
            "data_source": "YouTube API (real-time) + BigQuery benchmark",
        }

    def print_report(self) -> None:
        d = self.to_dict()
        vp = d["viral_potential"]
        km = d["key_metrics"]
        sep = "-" * 50
        print()
        print(sep)
        print(f"CHANNEL REPORT: {d['channel_name']}")
        print(sep)
        print(f"  Channel ID      : {d['channel_id']}")
        print(f"  Subscribers     : {_fmt_number(self.subscribers)}")
        print(f"  Tier            : {d['cluster']['name']}")
        print()
        print("  VIRAL POTENTIAL")
        print(f"  Probability     : {vp['probability']*100:.1f}%")
        print(f"  Label           : {vp['label']}")
        print(f"  Confidence      : {vp['confidence']}")
        print(f"  Rank in tier    : top {max(1, 100-vp['percentile_vs_benchmark'])}%")
        print()
        print("  KEY METRICS")
        print(f"  Avg views/video : {km['avg_views_per_video']}")
        print(f"  Like ratio      : {km['like_ratio']}")
        print(f"  Comment ratio   : {km['comment_ratio']}")
        print(f"  Recent trend    : {d['recent_trend']}")
        print()
        print(f"  Summary: {d['summary']}")
        if d["risk_factors"]:
            print()
            print("  Risks:")
            for r in d["risk_factors"]:
                print(f"    - {r}")
        print(sep)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def _cluster_desc(self) -> str:
        templates = {
            "Tier 0": "Kênh siêu lớn với lượng subscriber và views khổng lồ.",
            "Tier 1": "Kênh lớn có tầm ảnh hưởng cao và engagement tốt.",
            "Tier 2": "Kênh đang phát triển với audience trung thành.",
            "Tier 3": "Kênh niche với nội dung chuyên biệt.",
            "Tier 4": "Kênh nhỏ đang xây dựng audience.",
        }
        for key, desc in templates.items():
            if key in self.cluster_name:
                return desc
        return "Nhóm kênh có đặc điểm tương đồng."


# ─── Video Report ──────────────────────────────────────────────────────────────
@dataclass
class VideoReport:
    video_id: str
    video_title: str
    channel_name: str
    published_at: str
    video_age: str
    prediction: dict
    current_views: int
    views_per_hour: float
    vs_channel_avg_pct: float
    channel_percentile: int
    explanation: dict
    projected_views: dict

    def to_dict(self) -> dict:
        pred = self.prediction
        has_early = pred.get("has_early_signals", False)
        vs_str = (f"+{self.vs_channel_avg_pct:.0f}%" if self.vs_channel_avg_pct >= 0
                  else f"{self.vs_channel_avg_pct:.0f}%")

        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "channel": self.channel_name,
            "published_at": self.published_at,
            "video_age": self.video_age,

            "viral_prediction": {
                "will_viral": pred.get("will_viral"),
                "probability": pred.get("probability"),
                "time_window": pred.get("time_window"),
                "label": pred.get("label"),
                "confidence": pred.get("confidence"),
            },

            "current_performance": {
                "views": _fmt_number(self.current_views),
                "views_per_hour": _fmt_number(int(self.views_per_hour)),
                "vs_channel_avg": vs_str,
                "channel_percentile": self.channel_percentile,
            },

            "early_signals": {
                "available": has_early,
                "trend": self._trend_label(),
                "momentum_score": self.explanation.get("momentum_score", 50),
            },

            "explanation": [f["description"] for f in self.explanation.get("factors", [])],
            "summary": self.explanation.get("summary", ""),
            "projected_views": self.projected_views,
            "warnings": self.explanation.get("warnings", []),
            "model_used": "early_signal_model" if has_early else "snapshot_model",
        }

    def print_report(self) -> None:
        d = self.to_dict()
        vp = d["viral_prediction"]
        cp = d["current_performance"]
        es = d["early_signals"]
        pv = d["projected_views"]
        sep = "-" * 50
        print()
        print(sep)
        print("VIDEO REPORT")
        print(sep)
        print(f"  Title       : {self.video_title[:60]}")
        print(f"  Channel     : {self.channel_name}")
        print(f"  Published   : {self.published_at}")
        print(f"  Age         : {self.video_age}")
        print()
        print("  VIRAL PREDICTION")
        print(f"  Label       : {vp['label']}")
        print(f"  Probability : {vp['probability']*100:.1f}%")
        print(f"  Time window : {vp['time_window']}")
        print(f"  Confidence  : {vp['confidence']}")
        print()
        print("  CURRENT PERFORMANCE")
        print(f"  Views       : {cp['views']}")
        print(f"  Views/hour  : {cp['views_per_hour']}")
        print(f"  vs Pace     : {cp['vs_channel_avg']}")
        print(f"  Percentile  : top {100-self.channel_percentile}%")
        print(f"  Early signals: {'yes' if es['available'] else 'no'}")
        print(f"  Momentum    : {es['momentum_score']}/100")
        print()
        print("  PROJECTED VIEWS")
        print(f"  7 days      : {pv.get('7_days', 'N/A')}")
        print(f"  30 days     : {pv.get('30_days', 'N/A')}")
        print(f"  Confidence  : {pv.get('confidence', 'N/A')}")
        print()
        print(f"  Summary: {d['summary']}")
        if d["warnings"]:
            print()
            print("  Warnings:")
            for w in d["warnings"]:
                print(f"    - {w}")
        print(sep)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def _trend_label(self) -> str:
        mom = self.explanation.get("momentum_score", 50)
        if mom >= 70:
            return "ACCELERATING"
        if mom >= 50:
            return "STABLE"
        return "SLOWING"
