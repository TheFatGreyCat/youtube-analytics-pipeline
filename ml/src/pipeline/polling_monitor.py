"""
Polling Monitor — theo dõi video mới đăng theo background thread.
Poll API mỗi N giờ trong T giờ đầu, lưu snapshots vào JSON cache.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

POLL_DATA_DIR = Path(__file__).parent.parent.parent / "cache" / "polls"


class PollingMonitor:
    """
    Background monitor cho video mới đăng.

    Usage:
        monitor = PollingMonitor(api_client)
        monitor.start(video_id="abc123", interval_hours=6, duration_hours=72)
        # Sau đó có thể lấy snapshots:
        snapshots = monitor.get_snapshots("abc123")
    """

    def __init__(self, api_client) -> None:
        self._api = api_client
        self._threads: dict[str, threading.Thread] = {}
        self._stop_flags: dict[str, threading.Event] = {}
        POLL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        video_id: str,
        interval_hours: int = 6,
        duration_hours: int = 72,
        on_new_snapshot: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """
        Bắt đầu monitor video.

        Args:
            video_id:        ID video YouTube
            interval_hours:  Khoảng cách giữa 2 lần poll (giờ)
            duration_hours:  Tổng thời gian monitor (giờ)
            on_new_snapshot: Callback được gọi mỗi khi có snapshot mới
        """
        if video_id in self._threads and self._threads[video_id].is_alive():
            logger.warning("Video %s đang được monitor rồi.", video_id)
            return

        stop_flag = threading.Event()
        self._stop_flags[video_id] = stop_flag

        # Khởi tạo file dữ liệu
        poll_path = self._poll_path(video_id)
        if not poll_path.exists():
            self._save_poll_data(video_id, {
                "video_id": video_id,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "interval_hours": interval_hours,
                "duration_hours": duration_hours,
                "snapshots": [],
            })

        thread = threading.Thread(
            target=self._monitor_loop,
            args=(video_id, interval_hours, duration_hours, stop_flag, on_new_snapshot),
            daemon=True,
            name=f"poll-{video_id[:8]}",
        )
        self._threads[video_id] = thread
        thread.start()
        logger.info(
            "⏱  Bắt đầu monitor video %s (mỗi %dh, trong %dh)",
            video_id, interval_hours, duration_hours,
        )

    def stop(self, video_id: str) -> None:
        """Dừng monitor video."""
        flag = self._stop_flags.get(video_id)
        if flag:
            flag.set()
            logger.info("🛑 Dừng monitor: %s", video_id)

    def stop_all(self) -> None:
        for vid in list(self._stop_flags.keys()):
            self.stop(vid)

    def get_snapshots(self, video_id: str) -> list[dict]:
        """Lấy tất cả snapshots đã lưu cho video."""
        data = self._load_poll_data(video_id)
        return data.get("snapshots", [])

    def is_monitoring(self, video_id: str) -> bool:
        thread = self._threads.get(video_id)
        return thread is not None and thread.is_alive()

    # ── Private ────────────────────────────────────────────────────────────────
    def _monitor_loop(
        self,
        video_id: str,
        interval_hours: int,
        duration_hours: int,
        stop_flag: threading.Event,
        on_new_snapshot: Optional[Callable[[dict], None]],
    ) -> None:
        start = time.time()
        interval_secs = interval_hours * 3600
        duration_secs = duration_hours * 3600

        while not stop_flag.is_set():
            elapsed = time.time() - start
            if elapsed >= duration_secs:
                logger.info("✅ Monitor %s hoàn thành sau %dh", video_id, duration_hours)
                break

            # Fetch snapshot
            try:
                stats = self._api.get_video_stats([video_id])
                if video_id in stats:
                    snapshot = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        **stats[video_id],
                    }
                    self._append_snapshot(video_id, snapshot)
                    logger.info(
                        "📸 Snapshot %s: views=%d, likes=%d",
                        video_id, snapshot.get("views", 0), snapshot.get("likes", 0),
                    )
                    if on_new_snapshot:
                        on_new_snapshot(snapshot)
            except Exception as e:
                logger.error("❌ Lỗi khi poll %s: %s", video_id, e)

            # Chờ đến lần poll tiếp theo
            stop_flag.wait(timeout=interval_secs)

    def _poll_path(self, video_id: str) -> Path:
        return POLL_DATA_DIR / f"{video_id}.json"

    def _load_poll_data(self, video_id: str) -> dict:
        p = self._poll_path(video_id)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"video_id": video_id, "snapshots": []}

    def _save_poll_data(self, video_id: str, data: dict) -> None:
        p = self._poll_path(video_id)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_snapshot(self, video_id: str, snapshot: dict) -> None:
        data = self._load_poll_data(video_id)
        data.setdefault("snapshots", []).append(snapshot)
        self._save_poll_data(video_id, data)
