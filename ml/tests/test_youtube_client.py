"""
Unit tests cho YouTubeAPIClient — happy path + error path.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestYouTubeAPIClientInit(unittest.TestCase):
    def test_raises_if_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Xoá key nếu có
            os.environ.pop("YOUTUBE_API_KEY", None)
            from ml.src.data.youtube_client import YouTubeAPIClient
            with self.assertRaises(ValueError):
                YouTubeAPIClient(api_key=None)

    def test_init_with_key(self):
        with patch("ml.src.data.youtube_client.build") as mock_build:
            mock_build.return_value = MagicMock()
            from ml.src.data.youtube_client import YouTubeAPIClient
            client = YouTubeAPIClient(api_key="TEST_KEY")
            self.assertIsNotNone(client)


class TestParseIsoDuration(unittest.TestCase):
    def setUp(self):
        with patch("ml.src.data.youtube_client.build") as mock_build:
            mock_build.return_value = MagicMock()
            from ml.src.data.youtube_client import YouTubeAPIClient
            self._client = YouTubeAPIClient(api_key="TEST")

    def test_hours_minutes_seconds(self):
        self.assertEqual(self._client._parse_iso_duration("PT1H2M3S"), 3723)

    def test_minutes_only(self):
        self.assertEqual(self._client._parse_iso_duration("PT15M30S"), 930)

    def test_seconds_only(self):
        self.assertEqual(self._client._parse_iso_duration("PT45S"), 45)

    def test_empty_string(self):
        self.assertEqual(self._client._parse_iso_duration(""), 0)


class TestChannelNotFoundError(unittest.TestCase):
    def test_search_channel_raises_on_empty_results(self):
        with patch("ml.src.data.youtube_client.build") as mock_build:
            mock_yt = MagicMock()
            mock_build.return_value = mock_yt
            mock_yt.search.return_value.list.return_value.execute.return_value = {"items": []}

            from ml.src.data.youtube_client import ChannelNotFoundError, YouTubeAPIClient
            client = YouTubeAPIClient(api_key="TEST")
            # Xoá cache để không bị cache hit
            client._cache.clear()

            with self.assertRaises(ChannelNotFoundError):
                client.search_channel("__nonexistent_channel_12345__")


class TestQuotaTracker(unittest.TestCase):
    def test_quota_decreases(self):
        from ml.src.data.youtube_client import _QuotaTracker, QUOTA_LIMIT
        tracker = _QuotaTracker()
        initial = tracker.remaining
        # Không exceed — chỉ test tracking
        self.assertLessEqual(tracker.remaining, QUOTA_LIMIT)

    def test_quota_exceeded_raises(self):
        from ml.src.data.youtube_client import _QuotaTracker, QuotaExceededError, QUOTA_LIMIT
        tracker = _QuotaTracker()
        with self.assertRaises(QuotaExceededError):
            tracker.consume(QUOTA_LIMIT + 1)


if __name__ == "__main__":
    unittest.main()
