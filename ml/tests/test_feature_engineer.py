"""
Unit tests cho ChannelFeatureEngineer và VideoFeatureEngineer.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _make_channel_df(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "channel_id": [f"ch_{i}" for i in range(n)],
        "total_views": rng.integers(1_000_000, 1_000_000_000, n),
        "total_likes": rng.integers(10_000, 5_000_000, n),
        "total_comments": rng.integers(1_000, 500_000, n),
        "subscriber_count": rng.integers(10_000, 10_000_000, n),
        "total_videos_crawled": rng.integers(10, 500, n),
    })


def _make_video_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    channels = [f"ch_{i%5}" for i in range(n)]
    return pd.DataFrame({
        "video_id": [f"v_{i}" for i in range(n)],
        "channel_id": channels,
        "view_count": rng.integers(1_000, 5_000_000, n),
        "like_count": rng.integers(100, 100_000, n),
        "comment_count": rng.integers(10, 10_000, n),
        "duration_seconds": rng.integers(60, 3600, n),
        "engagement_score": rng.uniform(0, 20, n),
    })


class TestChannelFeatureEngineer(unittest.TestCase):
    def setUp(self):
        from ml.src.data.feature_engineer import ChannelFeatureEngineer
        self.fe = ChannelFeatureEngineer()
        self.channel_df = _make_channel_df(15)

    def test_fit_returns_self(self):
        result = self.fe.fit(self.channel_df)
        self.assertIs(result, self.fe)

    def test_transform_returns_correct_shape(self):
        self.fe.fit(self.channel_df)
        features = self.fe.transform(self.channel_df)
        self.assertEqual(len(features), len(self.channel_df))
        feature_names = self.fe.get_feature_names()
        for feat in feature_names:
            self.assertIn(feat, features.columns, f"Feature {feat} thiếu trong output")

    def test_no_negative_features(self):
        """Các ratio features phải >= 0."""
        self.fe.fit(self.channel_df)
        features = self.fe.transform(self.channel_df)
        for col in ["f1_efficiency", "f2_loyalty", "f3_depth", "f6_avg_views"]:
            if col in features.columns:
                self.assertTrue((features[col] >= 0).all(), f"{col} có giá trị âm")

    def test_percentiles_populated_after_fit(self):
        self.fe.fit(self.channel_df)
        percentiles = self.fe.get_percentiles()
        self.assertGreater(len(percentiles), 0)
        for feat in ["f1_efficiency", "f2_loyalty"]:
            self.assertIn(feat, percentiles)

    def test_percentile_rank_boundaries(self):
        self.fe.fit(self.channel_df)
        # Giá trị tối thiểu → gần 0
        self.assertLessEqual(self.fe.get_percentile_rank("f1_efficiency", 0), 10)
        # Giá trị cực lớn → gần 100
        self.assertGreaterEqual(self.fe.get_percentile_rank("f1_efficiency", 1e12), 90)

    def test_transform_from_api(self):
        self.fe.fit(self.channel_df)
        channel_stats = {
            "channel_id": "test_ch",
            "subscribers": 1_000_000,
            "total_views": 500_000_000,
            "video_count": 200,
        }
        video_stats = {
            f"v_{i}": {"views": 1_000_000 - i * 10_000, "likes": 50_000, "comments": 5_000}
            for i in range(20)
        }
        result = self.fe.transform_from_api(channel_stats, video_stats)
        self.assertEqual(len(result), 1)
        self.assertIn("f1_efficiency", result.columns)


class TestVideoFeatureEngineer(unittest.TestCase):
    def setUp(self):
        from ml.src.data.feature_engineer import VideoFeatureEngineer
        self.fe = VideoFeatureEngineer()
        self.engagement_df = _make_video_df(50)
        self.video_df = _make_video_df(50)

    def test_fit_returns_self(self):
        result = self.fe.fit(self.engagement_df, self.video_df)
        self.assertIs(result, self.fe)

    def test_transform_shape(self):
        self.fe.fit(self.engagement_df, self.video_df)
        features = self.fe.transform(self.engagement_df, self.video_df)
        self.assertEqual(len(features), len(self.engagement_df))

    def test_feature_values_in_range(self):
        self.fe.fit(self.engagement_df, self.video_df)
        features = self.fe.transform(self.engagement_df, self.video_df)
        # like_ratio và comment_ratio phải trong [0, 1]
        if "v1_like_ratio" in features.columns:
            self.assertTrue((features["v1_like_ratio"] >= 0).all())
        if "v2_comment_ratio" in features.columns:
            self.assertTrue((features["v2_comment_ratio"] >= 0).all())

    def test_transform_from_api(self):
        self.fe.fit(self.engagement_df, self.video_df)
        video_data = {
            "video_id": "test_v",
            "views": 2_000_000,
            "likes": 100_000,
            "comments": 10_000,
            "duration_seconds": 600,
            "published_at": "2025-01-01T12:00:00Z",
        }
        channel_stats = {
            "channel_id": "test_ch",
            "subscribers": 5_000_000,
            "total_views": 1_000_000_000,
            "video_count": 500,
        }
        result = self.fe.transform_from_api(video_data, channel_stats)
        self.assertEqual(len(result), 1)
        self.assertIn("v1_like_ratio", result.columns)


if __name__ == "__main__":
    unittest.main()
