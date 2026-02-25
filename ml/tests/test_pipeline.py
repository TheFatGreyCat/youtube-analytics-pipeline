"""
Unit tests cho pipeline: report_generator, polling_monitor, và explainer.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


# ─── Report Generator ──────────────────────────────────────────────────────────
class TestChannelReport(unittest.TestCase):
    def _make_report(self):
        from ml.src.pipeline.report_generator import ChannelReport
        return ChannelReport(
            input_name="MrBeast",
            channel_id="UCX6OQ3DkcsbYNE6H8uQQuVA",
            channel_name="MrBeast",
            subscribers=250_000_000,
            probability=0.94,
            confidence="HIGH",
            cluster_id=0,
            cluster_name="Tier 0 — Mega Channels",
            explanation={
                "summary": "Test summary.",
                "factors": [{"feature": "f1_efficiency", "impact": "positive", "description": "Test"}],
                "risk_factors": [],
                "strengths": ["Test strength"],
            },
            avg_views_per_video=45_000_000,
            like_ratio=0.042,
            comment_ratio=0.008,
            recent_trend=1.2,
            percentile_vs_benchmark=98.0,
        )

    def test_to_dict_structure(self):
        report = self._make_report()
        d = report.to_dict()
        self.assertIn("channel_name", d)
        self.assertIn("viral_potential", d)
        self.assertIn("cluster", d)
        self.assertIn("key_metrics", d)
        self.assertEqual(d["channel_name"], "MrBeast")

    def test_probability_in_dict(self):
        report = self._make_report()
        d = report.to_dict()
        self.assertEqual(d["viral_potential"]["probability"], 0.94)

    def test_to_json_valid(self):
        import json
        report = self._make_report()
        json_str = report.to_json()
        parsed = json.loads(json_str)
        self.assertIn("channel_name", parsed)

    def test_viral_label_high(self):
        from ml.src.pipeline.report_generator import _viral_label
        self.assertEqual(_viral_label(0.90), "🔥 EXTREMELY HIGH")
        self.assertEqual(_viral_label(0.72), "📈 HIGH")
        self.assertEqual(_viral_label(0.57), "⚡ MEDIUM")
        self.assertEqual(_viral_label(0.42), "⚠️ LOW")
        self.assertEqual(_viral_label(0.20), "❌ VERY LOW")


class TestVideoReport(unittest.TestCase):
    def _make_report(self):
        from ml.src.pipeline.report_generator import VideoReport
        return VideoReport(
            video_id="dQw4w9WgXcQ",
            video_title="MrBeast giveaway 2025",
            channel_name="MrBeast",
            published_at="2025-01-01T18:00:00Z",
            video_age="2 ngày 6 giờ",
            prediction={
                "will_viral": True,
                "probability": 0.83,
                "time_window": "viral_within_7d",
                "label": "⚡ VIRAL TRONG TUẦN",
                "confidence": "MEDIUM",
                "has_early_signals": True,
            },
            current_views=2_340_123,
            views_per_hour=43_521,
            vs_channel_avg_pct=340.0,
            channel_percentile=92,
            explanation={
                "summary": "Video đang viral",
                "factors": [{"description": "Like ratio cao"}],
                "warnings": [],
                "momentum_score": 78,
            },
            projected_views={"7_days": "~18M", "30_days": "~35M", "confidence": "MEDIUM"},
        )

    def test_to_dict_structure(self):
        report = self._make_report()
        d = report.to_dict()
        self.assertIn("viral_prediction", d)
        self.assertIn("current_performance", d)
        self.assertIn("early_signals", d)
        self.assertIn("projected_views", d)

    def test_will_viral_true(self):
        report = self._make_report()
        d = report.to_dict()
        self.assertTrue(d["viral_prediction"]["will_viral"])

    def test_to_json_valid(self):
        import json
        report = self._make_report()
        parsed = json.loads(report.to_json())
        self.assertEqual(parsed["channel"], "MrBeast")


# ─── Explainer ─────────────────────────────────────────────────────────────────
class TestPredictionExplainer(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import pandas as pd
        from ml.src.models.explainer import PredictionExplainer
        rng = np.random.default_rng(42)
        self.explainer = PredictionExplainer()
        # Fit với mock percentiles
        mock_percentiles = {
            "f1_efficiency": {"min": 0.5, "p25": 2.0, "p50": 5.0, "p75": 12.0, "max": 100.0, "mean": 6.0, "std": 3.0},
            "f2_loyalty": {"min": 0.001, "p25": 0.02, "p50": 0.04, "p75": 0.06, "max": 0.15, "mean": 0.04, "std": 0.02},
        }
        self.explainer.fit(channel_fe_percentiles=mock_percentiles)

    def test_explain_channel_returns_keys(self):
        import pandas as pd
        features = pd.DataFrame([{
            "f1_efficiency": 15.0, "f2_loyalty": 0.07, "f3_depth": 0.005,
            "f4_consistency": 0.8, "f6_avg_views": 5_000_000,
            "f7_engagement": 0.06, "f9_sub_tier": 7.5, "f11_recent_trend": 1.2,
        }])
        result = self.explainer.explain_channel(features, 0.85)
        self.assertIn("summary", result)
        self.assertIn("factors", result)
        self.assertIn("risk_factors", result)

    def test_project_views_reasonable(self):
        from ml.src.models.explainer import PredictionExplainer
        projected = PredictionExplainer.project_views(
            current_views=1_000_000,
            views_per_hour=50_000,
            probability=0.8,
            time_window="viral_within_7d",
        )
        self.assertIn("7_days", projected)
        self.assertIn("30_days", projected)


# ─── Polling Monitor ───────────────────────────────────────────────────────────
class TestPollingMonitor(unittest.TestCase):
    def test_start_and_stop(self):
        import time
        from ml.src.pipeline.polling_monitor import PollingMonitor
        mock_api = MagicMock()
        mock_api.get_video_stats.return_value = {
            "test_vid": {"views": 100_000, "likes": 5_000, "comments": 500}
        }
        monitor = PollingMonitor(mock_api)
        monitor.start("test_vid", interval_hours=0.001, duration_hours=0.01)
        time.sleep(0.1)
        self.assertTrue(monitor.is_monitoring("test_vid"))
        monitor.stop("test_vid")
        time.sleep(0.1)

    def test_get_snapshots_empty_initially(self):
        from ml.src.pipeline.polling_monitor import PollingMonitor
        mock_api = MagicMock()
        monitor = PollingMonitor(mock_api)
        snapshots = monitor.get_snapshots("nonexistent_video")
        self.assertEqual(snapshots, [])


if __name__ == "__main__":
    unittest.main()
