"""
Unit tests cho Labels, Clustering, Model A và Model B.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _make_channel_df(n: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "channel_id": [f"ch_{i}" for i in range(n)],
        "total_views": rng.integers(10_000_000, 10_000_000_000, n),
        "total_likes": rng.integers(100_000, 50_000_000, n),
        "total_comments": rng.integers(10_000, 5_000_000, n),
        "subscriber_count": rng.integers(100_000, 50_000_000, n),
        "total_videos_crawled": rng.integers(20, 500, n),
    })


def _make_video_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "video_id": [f"v_{i}" for i in range(n)],
        "channel_id": [f"ch_{i%10}" for i in range(n)],
        "view_count": rng.integers(1_000, 10_000_000, n),
        "like_count": rng.integers(100, 200_000, n),
        "comment_count": rng.integers(10, 20_000, n),
        "duration_seconds": rng.integers(60, 3_600, n),
        "engagement_score": rng.uniform(0, 20, n),
    })


# ─── Label Creator ─────────────────────────────────────────────────────────────
class TestChannelLabelCreator(unittest.TestCase):
    def setUp(self):
        from ml.src.models.label_creator import ChannelLabelCreator
        self.creator = ChannelLabelCreator()
        self.channel_df = _make_channel_df(42)

    def test_creates_viral_column(self):
        result = self.creator.create_labels(self.channel_df)
        self.assertIn("is_viral_channel", result.columns)

    def test_labels_are_binary(self):
        result = self.creator.create_labels(self.channel_df)
        unique_vals = set(result["is_viral_channel"].unique())
        self.assertTrue(unique_vals.issubset({0, 1}))

    def test_at_least_some_positives(self):
        result = self.creator.create_labels(self.channel_df)
        self.assertGreater(result["is_viral_channel"].sum(), 0)

    def test_derived_metrics_non_negative(self):
        result = self.creator.create_labels(self.channel_df)
        for col in ["efficiency_ratio", "loyalty_ratio", "depth_ratio", "avg_views_per_video"]:
            self.assertTrue((result[col] >= 0).all(), f"{col} có giá trị âm")

    def test_thresholds_accessible(self):
        self.creator.create_labels(self.channel_df)
        thresholds = self.creator.get_thresholds()
        self.assertIn("efficiency_p75", thresholds)


class TestVideoLabelCreator(unittest.TestCase):
    def setUp(self):
        from ml.src.models.label_creator import VideoLabelCreator
        self.creator = VideoLabelCreator()
        self.video_df = _make_video_df(200)

    def test_creates_is_viral_column(self):
        result = self.creator.create_labels(self.video_df)
        self.assertIn("is_viral", result.columns)

    def test_labels_binary(self):
        result = self.creator.create_labels(self.video_df)
        unique_vals = set(result["is_viral"].unique())
        self.assertTrue(unique_vals.issubset({0, 1}))

    def test_relative_score_present(self):
        result = self.creator.create_labels(self.video_df)
        self.assertIn("relative_score", result.columns)


# ─── Channel Clusterer ─────────────────────────────────────────────────────────
class TestChannelClusterer(unittest.TestCase):
    def setUp(self):
        from ml.src.data.feature_engineer import ChannelFeatureEngineer
        from ml.src.models.channel_clusterer import ChannelClusterer
        channel_df = _make_channel_df(42)
        self.fe = ChannelFeatureEngineer()
        self.fe.fit(channel_df)
        self.features = self.fe.transform(channel_df)
        self.clusterer = ChannelClusterer()

    def test_fit_without_autofind(self):
        self.clusterer.n_clusters = 4
        self.clusterer.fit(self.features, auto_find_k=False)
        self.assertTrue(self.clusterer._is_fitted)

    def test_assign_cluster_returns_valid(self):
        self.clusterer.n_clusters = 4
        self.clusterer.fit(self.features, auto_find_k=False)
        cluster_id, distance, stats = self.clusterer.assign_cluster(self.features.iloc[[0]])
        self.assertIn(cluster_id, range(4))
        self.assertGreaterEqual(distance, 0)

    def test_cluster_names_populated(self):
        self.clusterer.n_clusters = 3
        self.clusterer.fit(self.features, auto_find_k=False)
        for cid in range(3):
            name = self.clusterer.get_cluster_name(cid)
            self.assertIsInstance(name, str)


# ─── Model A ───────────────────────────────────────────────────────────────────
class TestChannelViralClassifier(unittest.TestCase):
    def setUp(self):
        from ml.src.data.feature_engineer import ChannelFeatureEngineer
        from ml.src.models.channel_classifier import ChannelViralClassifier
        from ml.src.models.label_creator import ChannelLabelCreator
        channel_df = _make_channel_df(30)
        labeled = ChannelLabelCreator().create_labels(channel_df)
        fe = ChannelFeatureEngineer()
        fe.fit(labeled)
        features = fe.transform(labeled)
        features["is_viral_channel"] = labeled["is_viral_channel"].values
        self.features = features
        self.clf = ChannelViralClassifier()

    def test_train_returns_metrics(self):
        result = self.clf.train(self.features)
        self.assertIn("accuracy", result)
        self.assertIn("f1", result)

    def test_predict_proba_in_range(self):
        self.clf.train(self.features)
        prob, conf = self.clf.predict_proba(self.features.iloc[[0]])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertIn(conf, ["HIGH", "MEDIUM", "LOW"])

    def test_feature_importances_keys(self):
        self.clf.train(self.features)
        fi = self.clf.get_feature_importances()
        self.assertIsInstance(fi, dict)

    def test_predict_without_train_raises(self):
        from ml.src.models.channel_classifier import ChannelViralClassifier
        clf = ChannelViralClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict_proba(self.features.iloc[[0]])


# ─── Model B ───────────────────────────────────────────────────────────────────
class TestVideoViralClassifier(unittest.TestCase):
    def setUp(self):
        from ml.src.data.feature_engineer import VideoFeatureEngineer
        from ml.src.models.label_creator import VideoLabelCreator
        from ml.src.models.video_classifier import VideoViralClassifier
        video_df = _make_video_df(200)
        labeled = VideoLabelCreator().create_labels(video_df)
        fe = VideoFeatureEngineer()
        fe.fit(video_df, video_df)
        features = fe.transform(video_df, video_df)
        features["is_viral"] = labeled["is_viral"].values[:len(features)]
        if "time_window_label" in labeled.columns:
            features["time_window_label"] = labeled["time_window_label"].values[:len(features)]
        self.features = features
        self.clf = VideoViralClassifier()

    def test_train_returns_metrics(self):
        result = self.clf.train(self.features)
        self.assertIn("b1", result)
        self.assertIn("b2", result)

    def test_predict_output_structure(self):
        self.clf.train(self.features)
        result = self.clf.predict(self.features.iloc[[0]])
        self.assertIn("will_viral", result)
        self.assertIn("probability", result)
        self.assertIn("time_window", result)
        self.assertIn("confidence", result)
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)

    def test_predict_without_train_raises(self):
        from ml.src.models.video_classifier import VideoViralClassifier
        clf = VideoViralClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict(self.features.iloc[[0]])


if __name__ == "__main__":
    unittest.main()
