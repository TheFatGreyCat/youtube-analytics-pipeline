"""
Interactive terminal test for the YouTube Viral Video Prediction pipeline.

Nhap ten kenh YouTube → tim kenh → lay video moi nhat → du doan viral → hien thi ket qua.

Cach chay:1
    python test_predict.py
    python test_predict.py --verbose    <- hien thi INFO logs
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

# ── Logging setup ─────────────────────────────────────────────────────────────
_VERBOSE = "--verbose" in sys.argv
logging.basicConfig(
    level=logging.INFO if _VERBOSE else logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# HELPER — GỌI YOUTUBE API
# ═════════════════════════════════════════════════════════════════════════════

def _search_channels_multi(youtube_client, channel_name: str, max_results: int = 5) -> list[dict]:
    """
    Tim kiem tren YouTube va tra ve danh sach cac kenh khop.

    Moi phan tu trong list la dict chua:
      channel_id, channel_name, subscriber_count (str), video_count (str),
      published_at (ISO8601 — ngay tao kenh), country, description, ...

    Returns empty list neu khong tim thay hoac loi API.
    """
    from googleapiclient.errors import HttpError

    try:
        response = youtube_client.search().list(
            part="snippet",
            q=channel_name,
            type="channel",
            maxResults=max_results,
        ).execute()
    except HttpError:
        raise
    except Exception as exc:
        logger.warning("_search_channels_multi: %s", exc)
        return []

    items = response.get("items", [])
    if not items:
        return []

    channel_ids = [item["snippet"]["channelId"] for item in items]
    try:
        details_resp = youtube_client.channels().list(
            part="snippet,statistics",
            id=",".join(channel_ids),
        ).execute()
    except HttpError:
        raise
    except Exception as exc:
        logger.warning("channels.list batch: %s", exc)
        return []

    results: list[dict] = []
    for item in details_resp.get("items", []):
        snippet    = item.get("snippet", {})
        statistics = item.get("statistics", {})
        results.append({
            "channel_id":       item["id"],
            "channel_name":     snippet.get("title", ""),
            "description":      snippet.get("description", "")[:200],
            "custom_url":       snippet.get("customUrl", ""),
            "published_at":     snippet.get("publishedAt", ""),
            "country":          snippet.get("country", ""),
            "subscriber_count": statistics.get("subscriberCount", "0"),
            "video_count":      statistics.get("videoCount", "0"),
            "view_count":       statistics.get("viewCount", "0"),
        })

    return results


def _get_latest_video(youtube_client, channel_id: str) -> Optional[dict]:
    """
    Lay video moi nhat cua kenh tu uploads playlist.

    Returns:
        Raw YouTube API video item (snippet + statistics + contentDetails + status)
        hoac None neu kenh khong co video.
    """
    from googleapiclient.errors import HttpError

    try:
        ch_resp = youtube_client.channels().list(
            part="contentDetails",
            id=channel_id,
        ).execute()
    except HttpError:
        raise

    items = ch_resp.get("items", [])
    if not items:
        return None

    playlist_id = (
        items[0]
        .get("contentDetails", {})
        .get("relatedPlaylists", {})
        .get("uploads", "")
    )
    if not playlist_id:
        return None

    try:
        pl_resp = youtube_client.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=1,
        ).execute()
    except HttpError:
        raise

    pl_items = pl_resp.get("items", [])
    if not pl_items:
        return None

    video_id = pl_items[0]["snippet"]["resourceId"]["videoId"]

    try:
        v_resp = youtube_client.videos().list(
            part="snippet,statistics,contentDetails,status",
            id=video_id,
        ).execute()
    except HttpError:
        raise

    v_items = v_resp.get("items", [])
    return v_items[0] if v_items else None


def _build_video_data(raw_item: dict) -> dict:
    """
    Chuyen tu YouTube API video item sang dict ma predict_from_api_data() nhan.
    """
    snippet         = raw_item.get("snippet", {})
    statistics      = raw_item.get("statistics", {})
    content_details = raw_item.get("contentDetails", {})
    status          = raw_item.get("status", {})

    return {
        "video_id":         raw_item["id"],
        "title":            snippet.get("title", ""),
        "description":      snippet.get("description", ""),
        "tags":             snippet.get("tags", []),
        "category_id":      snippet.get("categoryId", "unknown"),
        "default_language": snippet.get("defaultLanguage") or snippet.get("defaultAudioLanguage"),
        "published_at":     snippet.get("publishedAt", ""),
        "duration_iso8601": content_details.get("duration", ""),
        "has_caption":      content_details.get("caption", "false").lower() == "true",
        "definition":       content_details.get("definition", "hd"),
        "is_embeddable":    status.get("embeddable", True),
        "is_made_for_kids": status.get("madeForKids", False),
        "view_count":       int(statistics.get("viewCount")  or 0),
        "like_count":       int(statistics.get("likeCount")  or 0),
        "comment_count":    int(statistics.get("commentCount") or 0),
    }


def _build_channel_data(details: dict) -> dict:
    """
    Chuyen tu ChannelFinder.get_channel_details() sang dict ma predict_from_api_data() nhan.
    """
    return {
        "channel_id":        details["channel_id"],
        "channel_name":      details.get("channel_name", ""),
        "subscriber_count":  int(details.get("subscriber_count") or 0),
        "total_video_count": int(details.get("video_count") or 0),
        "channel_created_at": details.get("published_at", ""),
        "country_code":      details.get("country", ""),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 2 — CHỌN KÊNH KHI TÌM THẤY NHIỀU KẾT QUẢ
# ═════════════════════════════════════════════════════════════════════════════

def _select_channel(channels: list[dict]) -> Optional[dict]:
    """
    Hien thi danh sach kenh va yeu cau nguoi dung chon.

    Returns:
        channel dict neu chon hop le, None neu chon 0 hoac nhap sai > 3 lan.
    """
    n    = len(channels)
    dash = "-" * 60

    print(f"\n  Tim thay {n} kenh. Vui long chon kenh dung:")
    print(f"  {dash}")

    for i, ch in enumerate(channels, start=1):
        try:
            subs = f"{int(ch.get('subscriber_count', 0)):,}"
        except (ValueError, TypeError):
            subs = "An"
        country = ch.get("country") or "Khong ro"
        print(f"  [{i}] {ch.get('channel_name', '')}")
        print(f"      Subscribers : {subs}")
        print(f"      Quoc gia    : {country}")
        print(f"      Channel ID  : {ch.get('channel_id', '')}")
        print()

    print("  [0] Khong phai kenh nao ca — thu lai")
    print(f"  {dash}")

    for attempt in range(3):
        try:
            raw = input("  Nhap so thu tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw == "0":
            return None

        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= n:
                return channels[idx - 1]

        remaining = 2 - attempt
        if remaining > 0:
            print(f"  Lua chon khong hop le. Con {remaining} lan thu.")
        else:
            print("  Da nhap sai 3 lan. Quay lai.")

    return None


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 1 — VÒNG LẶP CHÍNH
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Vong lap chinh cua chuong trinh du doan viral YouTube."""
    from googleapiclient.errors import HttpError

    from extract.channel_finder import ChannelFinder
    from ml.predict import format_prediction_output, predict_from_api_data
    from ml.save_load import list_saved_models, load_model

    sep = "=" * 60

    # ── KHỞI ĐỘNG ─────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  YouTube Viral Video Prediction")
    print(sep)
    print('  Nhap "exit"   de thoat.')
    print('  Nhap "models" de xem danh sach model da luu.')
    print()

    # Load model 1 lan duy nhat
    print("  Dang load model du doan ...")
    try:
        model, config = load_model()
        print("  Model san sang.\n")
    except FileNotFoundError:
        print("\n  [LOI] Model chua duoc train.")
        print("  Chay lenh sau de train: python -m ml.train")
        sys.exit(1)

    # Khoi tao YouTube API client thong qua ChannelFinder
    try:
        finder = ChannelFinder()
    except ValueError:
        print("\n  [LOI API] API key khong hop le.")
        print("  Kiem tra YOUTUBE_API_KEY trong file .env")
        sys.exit(1)
    except Exception as exc:
        print(f"\n  [LOI] Khong the khoi tao YouTube client: {exc}")
        sys.exit(1)

    youtube_client = finder.youtube

    # ── VÒNG LẶP ─────────────────────────────────────────────────────────────
    while True:

        # BƯỚC 1 — Nhận input ─────────────────────────────────────────────────
        try:
            channel_query = input("\nNhap ten kenh YouTube: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDa thoat.")
            break

        if channel_query.lower() == "exit":
            break

        if channel_query.lower() == "models":
            list_saved_models()
            continue

        if not channel_query:
            print("  Ten kenh khong duoc trong.")
            continue

        # BƯỚC 2 — Tìm kênh ───────────────────────────────────────────────────
        print(f"\n  Dang tim kiem kenh: {channel_query} ...")
        try:
            channels = _search_channels_multi(youtube_client, channel_query)
        except HttpError as exc:
            _handle_http_error(exc, channel_query)
            continue
        except Exception as exc:
            _handle_connection_error(exc)
            continue

        if not channels:
            print(f"  [KHONG TIM THAY] Khong tim thay kenh \"{channel_query}\".")
            print("  Goi y: Kiem tra chinh ta, thu ten day du hon.")
            continue

        if len(channels) == 1:
            print("  Tim thay 1 kenh.")
            channel_details = channels[0]
        else:
            print(f"  Tim thay {len(channels)} kenh.")
            channel_details = _select_channel(channels)
            if channel_details is None:
                continue

        channel_id   = channel_details["channel_id"]
        channel_name = channel_details.get("channel_name", channel_query)

        # BƯỚC 3 — Lấy video mới nhất ─────────────────────────────────────────
        print(f"\n  Dang lay du lieu kenh {channel_name} ...")
        try:
            raw_video = _get_latest_video(youtube_client, channel_id)
        except HttpError as exc:
            _handle_http_error(exc, channel_query)
            continue
        except Exception as exc:
            _handle_connection_error(exc)
            continue

        if raw_video is None:
            print(f"  [THONG BAO] Kenh \"{channel_name}\" chua co video nao.")
            continue

        video_data   = _build_video_data(raw_video)
        channel_data = _build_channel_data(channel_details)

        # BƯỚC 4 — Predict ────────────────────────────────────────────────────
        print("\n  Dang du doan ...")
        try:
            result = predict_from_api_data(video_data, channel_data)
        except Exception as exc:
            print(f"  [LOI] Da xay ra loi khong xac dinh: {exc}")
            logger.exception("predict_from_api_data failed")
            continue

        # BƯỚC 5 — Hiển thị kết quả ───────────────────────────────────────────
        print(format_prediction_output(result))

        # BƯỚC 6 — Hỏi tiếp ───────────────────────────────────────────────────
        try:
            choice = input("\nDu doan kenh khac? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nDa thoat.")
            break

        if choice != "y":
            break

    print("\nDa thoat.")


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 3 — XỬ LÝ EXCEPTION
# ═════════════════════════════════════════════════════════════════════════════

def _handle_http_error(exc, channel_name: str = "") -> None:
    """Phan loai va in thong bao loi HttpError tu YouTube API."""
    import json

    try:
        status  = int(exc.resp.status)
        content = json.loads(exc.content.decode("utf-8", errors="ignore"))
        errors  = content.get("error", {}).get("errors", [{}])
        reason  = errors[0].get("reason", "") if errors else ""
    except Exception:
        status = 0
        reason = ""

    quota_reasons = {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}

    if reason in quota_reasons or (status == 403 and "quota" in str(exc).lower()):
        print("\n  [LOI API] YouTube API da het quota hom nay.")
        print("  Vui long thu lai vao ngay mai.")
        sys.exit(1)
    elif reason == "keyInvalid" or (status == 400 and "key" in str(exc).lower()):
        print("\n  [LOI API] API key khong hop le.")
        print("  Kiem tra YOUTUBE_API_KEY trong file .env")
        sys.exit(1)
    elif status in (0, 500, 503):
        print("\n  [LOI KET NOI] Khong the ket noi YouTube API.")
        print("  Kiem tra ket noi internet va thu lai.")
    else:
        if channel_name:
            print(f"\n  [KHONG TIM THAY] Khong tim thay kenh \"{channel_name}\".")
            print("  Goi y: Kiem tra chinh ta, thu ten day du hon.")
        else:
            print(f"\n  [LOI API] {exc}")


def _handle_connection_error(exc: Exception) -> None:
    """In thong bao loi ket noi mang."""
    import socket

    if isinstance(exc, (ConnectionError, socket.timeout, TimeoutError)):
        print("\n  [LOI KET NOI] Khong the ket noi YouTube API.")
        print("  Kiem tra ket noi internet va thu lai.")
    else:
        print(f"\n  [LOI] Da xay ra loi khong xac dinh: {exc}")
        logger.exception("Unexpected error")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDa thoat.")
        sys.exit(0)
