"""
Prediction engine for the YouTube viral video prediction pipeline.

Loads the trained XGBoost model, prepares raw YouTube API data into
the exact feature matrix used during training, runs inference, and
returns a structured result with SHAP-based explanations.

Pipeline order:1
    train.py / save_load.py  →  predict.py

Usage (standalone):
    python -m ml.predict
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap

from ml.features import engineer_features, fill_missing
from ml.save_load import load_model

# ── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Feature → Vietnamese description mapping ─────────────────────────────────
FEATURE_MEANING: dict[str, str] = {
    "velocity_score":           "Toc do tang view so voi subscriber",
    "view_vs_channel_avg":      "Luot xem so voi trung binh kenh",
    "engagement_score":         "Muc do tuong tac tong the",
    "like_rate_pct":            "Ti le like tren luot xem",
    "comment_rate_pct":         "Ti le comment tren luot xem",
    "avg_views_per_day_log":    "Toc do tang view trung binh moi ngay",
    "like_rate_vs_channel":     "Ti le like so voi trung binh kenh",
    "comment_rate_vs_channel":  "Ti le comment so voi trung binh kenh",
    "engagement_level_enc":     "Cap do tuong tac (cao/trung binh/thap)",
    "subscriber_log":           "Quy mo kenh (so subscriber)",
    "channel_avg_views_log":    "Hieu suat trung binh cua kenh",
    "is_prime_time":            "Dang video trong gio vang (18-22h)",
    "is_weekend":               "Dang video vao cuoi tuan",
    "published_hour":           "Gio dang video",
    "published_dayofweek":      "Thu trong tuan dang video",
    "published_month":          "Thang dang video",
    "publish_quarter":          "Quy dang video",
    "title_length":             "Do dai tieu de (so ky tu)",
    "title_word_count":         "Do dai tieu de (so tu)",
    "has_number":               "Tieu de co chua so",
    "has_question":             "Tieu de co cau hoi (?)",
    "has_exclamation":          "Tieu de co dau cham than (!)",
    "has_emoji":                "Tieu de co emoji",
    "has_caps_word":            "Tieu de co tu viet hoa",
    "duration_minutes":         "Thoi luong video (phut)",
    "is_shorts":                "Dinh dang YouTube Shorts (<= 60s)",
    "is_short":                 "Video ngan (<= 10 phut)",
    "is_medium":                "Video vua (10-30 phut)",
    "tag_count":                "So luong tag",
    "has_tags":                 "Co su dung tag",
    "is_hd":                    "Video chat luong HD",
    "has_caption":              "Co phu de / transcript",
    "is_embeddable":            "Cho phep nhung video",
    "is_made_for_kids":         "Noi dung danh cho tre em",
    "category_id_enc":          "Danh muc video",
    "channel_age_days":         "Tuoi cua kenh (so ngay)",
    "upload_freq_per_day":      "Tan suat dang video (moi ngay)",
    "channel_avg_like_rate":    "Ti le like trung binh cua kenh",
    "channel_avg_cmt_rate":     "Ti le comment trung binh cua kenh",
    "duration_vs_channel":      "Thoi luong so voi trung binh kenh",
    "total_videos_crawled":     "So video da thu thap cua kenh",
    "is_morning":               "Dang video buoi sang (6-10h)",
    "is_lunch_slot":            "Dang video gio trua (11-13h)",
}

# Friendly names for top_negative_driver recommendation
_FEATURE_FRIENDLY: dict[str, str] = {
    "velocity_score":        "toc do tang view",
    "view_vs_channel_avg":   "luot xem so voi kenh",
    "engagement_score":      "muc do tuong tac",
    "like_rate_pct":         "ti le like",
    "comment_rate_pct":      "ti le comment",
    "avg_views_per_day_log": "toc do tang view hang ngay",
    "subscriber_log":        "quy mo kenh",
    "duration_minutes":      "thoi luong video",
    "tag_count":             "so luong tag",
    "title_length":          "do dai tieu de",
    "is_prime_time":         "gio dang video (nen dang 18-22h)",
    "is_weekend":            "thoi diem cuoi tuan",
    "channel_age_days":      "tuoi kenh",
}


# ═════════════════════════════════════════════════════════════════════════════
# HELPER
# ═════════════════════════════════════════════════════════════════════════════

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Return a / b, or ``default`` if b is zero or NaN."""
    if b == 0 or b != b:   # second check catches NaN
        return default
    return a / b


def _parse_iso8601_duration(duration_str: str) -> int:
    """
    Convert ISO 8601 duration string to total seconds.

    Examples
    --------
    "PT30S"      → 30
    "PT2M"       → 120
    "PT1H2M3S"   → 3723
    "P1DT2H3M4S" → 93784
    """
    if not duration_str:
        return 0
    pattern = re.compile(
        r"P(?:(?P<days>\d+)D)?"
        r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?"
    )
    m = pattern.fullmatch(duration_str.strip())
    if not m:
        logger.warning("Cannot parse duration: '%s' — defaulting to 0 seconds.", duration_str)
        return 0
    days    = int(m.group("days")    or 0)
    hours   = int(m.group("hours")   or 0)
    minutes = int(m.group("minutes") or 0)
    seconds = int(m.group("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 1 — CHUẨN BỊ DỮ LIỆU TỪ API
# ═════════════════════════════════════════════════════════════════════════════

def prepare_input_from_api(
    video_data: dict,
    channel_data: dict,
    train_medians: dict,
) -> pd.DataFrame:
    """
    Convert YouTube API dicts into a one-row DataFrame matching the
    column structure expected by ``engineer_features()``.

    Channel-level aggregate stats (avg_views_per_video, avg_like_rate_pct,
    etc.) are not available from the API for new videos; they are filled with
    training-set medians and recorded in ``df.attrs["warnings"]``.

    Parameters
    ----------
    video_data : dict
        Fields from YouTube Videos API response (snippet + contentDetails
        + statistics).
    channel_data : dict
        Fields from YouTube Channels API response.
    train_medians : dict
        Median values computed during training — used to impute
        channel-summary columns unavailable from the API.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with all columns ready for ``engineer_features()``.

    Raises
    ------
    KeyError
        If a required field is missing from ``video_data`` or ``channel_data``.
    """
    required_video  = {"video_id", "title", "published_at", "duration_iso8601"}
    required_channel = {"channel_id", "channel_name", "channel_created_at"}
    for key in required_video:
        if key not in video_data:
            raise KeyError(f"video_data thieu truong bat buoc: '{key}'")
    for key in required_channel:
        if key not in channel_data:
            raise KeyError(f"channel_data thieu truong bat buoc: '{key}'")

    warnings: list[str] = []

    # ── BƯỚC 1 — Parse published_at ──────────────────────────────────────────
    published_at = pd.to_datetime(video_data["published_at"], utc=True)
    published_hour      = published_at.hour
    published_month     = published_at.month
    # Convert isoweekday (Mon=1..Sun=7) → BigQuery convention (Sun=1..Sat=7)
    iso_dow = published_at.isoweekday()  # Mon=1, Sun=7
    published_dayofweek = iso_dow % 7 + 1  # Mon→2, Tue→3, ..., Sun→1, Sat→7

    now_utc = pd.Timestamp.now(tz="UTC")
    days_since_published = max(1, (now_utc - published_at).days)

    # ── BƯỚC 2 — Parse duration ───────────────────────────────────────────────
    duration_seconds = _parse_iso8601_duration(video_data.get("duration_iso8601", ""))

    # ── BƯỚC 3 — Tags ─────────────────────────────────────────────────────────
    raw_tags = video_data.get("tags")
    if isinstance(raw_tags, list):
        tags_str = ",".join(raw_tags)
    elif isinstance(raw_tags, str):
        tags_str = raw_tags
    else:
        tags_str = ""

    # ── BƯỚC 4 — Engagement metrics ───────────────────────────────────────────
    view_count    = int(video_data.get("view_count")    or 0)
    like_count    = int(video_data.get("like_count")    or 0)
    comment_count = int(video_data.get("comment_count") or 0)

    like_rate_pct     = safe_divide(like_count, view_count) * 100
    comment_rate_pct  = safe_divide(comment_count, view_count) * 100
    engagement_score  = safe_divide(like_count + comment_count * 2, view_count) * 100
    avg_views_per_day = safe_divide(view_count, days_since_published)

    if like_rate_pct >= 5.0:
        engagement_level = "high"
    elif like_rate_pct >= 2.0:
        engagement_level = "medium"
    else:
        engagement_level = "low"

    # ── BƯỚC 5 — video_length_category ───────────────────────────────────────
    if duration_seconds <= 60:
        video_length_category = "shorts"
    elif duration_seconds <= 600:
        video_length_category = "short"
    elif duration_seconds <= 1800:
        video_length_category = "medium"
    else:
        video_length_category = "long"

    # ── BƯỚC 6 — Channel context ──────────────────────────────────────────────
    subscriber_count = int(channel_data.get("subscriber_count") or 0)
    channel_created_at = pd.to_datetime(channel_data["channel_created_at"], utc=True)

    # Channel-summary aggregates: fill from train_medians with warnings
    channel_summary_cols = [
        "avg_views_per_video",
        "avg_like_rate_pct",
        "avg_comment_rate_pct",
        "avg_video_duration_seconds",
        "avg_days_between_uploads",
        "total_videos_crawled",
    ]
    channel_summary: dict[str, float] = {}
    for col in channel_summary_cols:
        if col in train_medians:
            channel_summary[col] = train_medians[col]
        else:
            channel_summary[col] = 0.0
        warnings.append(
            f"'{col}' khong co trong API — su dung gia tri train median ({channel_summary[col]:.4f})"
        )

    # ── BƯỚC 7 — Assemble DataFrame ───────────────────────────────────────────
    row = {
        # IDs
        "video_id":               video_data["video_id"],
        "channel_id":             channel_data["channel_id"],
        # Video metadata
        "title":                  video_data.get("title", ""),
        "description":            video_data.get("description", ""),
        "tags":                   tags_str,
        "category_id":            str(video_data.get("category_id", "unknown")),
        "default_language":       video_data.get("default_language"),
        "published_at":           published_at,
        "duration_iso8601":       video_data.get("duration_iso8601", ""),
        "duration_seconds":       duration_seconds,
        "has_caption":            int(bool(video_data.get("has_caption", False))),
        "definition":             video_data.get("definition", "hd"),
        "is_embeddable":          int(bool(video_data.get("is_embeddable", True))),
        "is_made_for_kids":       int(bool(video_data.get("is_made_for_kids", False))),
        # Temporal
        "published_hour":         published_hour,
        "published_month":        published_month,
        "published_dayofweek":    published_dayofweek,
        "published_year":         published_at.year,
        "days_since_published":   days_since_published,
        "video_length_category":  video_length_category,
        # Stats
        "view_count":             view_count,
        "like_count":             like_count,
        "comment_count":          comment_count,
        # Engagement metrics (mirrors int_engagement_metrics)
        "like_rate_pct":          like_rate_pct,
        "comment_rate_pct":       comment_rate_pct,
        "engagement_score":       engagement_score,
        "avg_views_per_day":      avg_views_per_day,
        "engagement_level":       engagement_level,
        # Channel info
        "channel_name":           channel_data.get("channel_name", ""),
        "channel_subscribers":    subscriber_count,
        "subscriber_count":       subscriber_count,
        "country_code":           channel_data.get("country_code"),
        "channel_created_at":     channel_created_at,
        # Channel summary aggregates (from train_medians)
        "avg_views_per_video":          channel_summary["avg_views_per_video"],
        "avg_like_rate_pct":            channel_summary["avg_like_rate_pct"],
        "avg_comment_rate_pct":         channel_summary["avg_comment_rate_pct"],
        "avg_video_duration_seconds":   channel_summary["avg_video_duration_seconds"],
        "avg_days_between_uploads":     channel_summary["avg_days_between_uploads"],
        "total_videos_crawled":         channel_summary["total_videos_crawled"],
        # Needed by interaction features
        "is_potentially_viral":  False,
    }

    df = pd.DataFrame([row])
    df.attrs["warnings"] = warnings
    return df


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 2 — HÀM CHÍNH PREDICT
# ═════════════════════════════════════════════════════════════════════════════

def predict_from_api_data(
    video_data: dict,
    channel_data: dict,
) -> dict:
    """
    Full prediction pipeline: load model → prepare input → engineer features
    → predict → explain with SHAP → format result.

    Parameters
    ----------
    video_data : dict
        Video fields from YouTube API (see module docstring for schema).
    channel_data : dict
        Channel fields from YouTube API.

    Returns
    -------
    dict
        Keys: video_id, channel_id, channel_name, title, published_at,
        days_since_published, view_count, viral_score, prediction,
        confidence, top_5_drivers, recommendation, warnings.

    Raises
    ------
    FileNotFoundError
        If the trained model files are not found.
    KeyError
        If required fields are missing from input dicts.
    """
    # ── BƯỚC 1 — Load model & config ─────────────────────────────────────────
    try:
        model, config = load_model()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Model chua duoc train. Chay 'python -m ml.train' truoc."
        ) from exc

    train_medians  = config.get("train_medians",  {})
    feature_list   = config.get("feature_list",   None)
    label_encoders = config.get("label_encoders", {})
    threshold      = float(config.get("viral_threshold", 0.5))

    # ── BƯỚC 2 — Chuẩn bị input ──────────────────────────────────────────────
    df = prepare_input_from_api(video_data, channel_data, train_medians)
    all_warnings: list[str] = list(df.attrs.get("warnings", []))
    days_since = int(df["days_since_published"].iloc[0])

    # ── BƯỚC 3 — Feature engineering ─────────────────────────────────────────
    X, _ = engineer_features(df, label_encoders=label_encoders, feature_list=feature_list)

    # ── BƯỚC 4 — Fill missing ─────────────────────────────────────────────────
    X, _ = fill_missing(X, train_medians=train_medians)

    # ── BƯỚC 5 — Align columns ───────────────────────────────────────────────
    if feature_list:
        for col in feature_list:
            if col not in X.columns:
                logger.warning("[CANH BAO] Feature '%s' thiếu — điền 0", col)
                X[col] = 0
        X = X[feature_list]

    # ── BƯỚC 6 — Predict ─────────────────────────────────────────────────────
    viral_score = float(model.predict_proba(X)[0, 1])
    is_viral    = viral_score >= threshold

    # ── BƯỚC 7 — SHAP explanation ─────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]   # class 1 (viral) for binary TreeExplainer (old API)
    else:
        sv = shap_values[0]       # new API: single 2D array

    shap_dict = dict(zip(X.columns, sv))
    top_5_sorted = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    top_5_drivers = [
        {
            "feature": feat,
            "impact":  round(float(impact), 4),
            "meaning": FEATURE_MEANING.get(feat, feat),
        }
        for feat, impact in top_5_sorted
    ]

    # ── BƯỚC 8 — Confidence ───────────────────────────────────────────────────
    if viral_score >= 0.80 or viral_score <= 0.20:
        confidence = "Cao"
    elif viral_score >= 0.65 or viral_score <= 0.35:
        confidence = "Trung binh"
    else:
        confidence = "Thap"

    # ── BƯỚC 9 — Recommendation ───────────────────────────────────────────────
    # Find top negative driver (largest negative SHAP impact)
    negative_drivers = [(f, v) for f, v in shap_dict.items() if v < 0]
    if negative_drivers:
        top_neg_feat = min(negative_drivers, key=lambda kv: kv[1])[0]
        top_neg_desc = _FEATURE_FRIENDLY.get(top_neg_feat, top_neg_feat)
    else:
        top_neg_desc = "features hien tai"

    if viral_score >= 0.75:
        recommendation = (
            "Video dang co da viral manh. "
            "Nen boost quang cao trong 24h toi."
        )
    elif viral_score >= 0.50:
        recommendation = (
            "Video co tiem nang. "
            "Theo doi them 24-48h truoc khi quyet dinh boost."
        )
    elif viral_score >= 0.30:
        recommendation = (
            f"Tiem nang viral thap. Can cai thien {top_neg_desc}."
        )
    else:
        recommendation = (
            "Video kho viral voi noi dung va thoi diem hien tai."
        )

    # ── BƯỚC 10 — Cảnh báo đặc biệt ─────────────────────────────────────────
    if days_since <= 1:
        all_warnings.append(
            "Video moi dang trong vong 24h. Du lieu engagement chua on dinh, "
            "ket qua du doan co the chua chinh xac."
        )
    if channel_data.get("subscriber_count") is None:
        all_warnings.append(
            "Kenh an so subscriber. Features lien quan duoc uoc tinh."
        )
    if video_data.get("like_count") is None:
        all_warnings.append(
            "Kenh an so like. Like rate duoc uoc tinh tu median training."
        )
    if int(video_data.get("view_count") or 0) == 0:
        all_warnings.append(
            "Video chua co luot xem. Engagement features = 0."
        )

    # ── Assemble result ───────────────────────────────────────────────────────
    pub_at_str = pd.to_datetime(video_data["published_at"], utc=True).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    return {
        "video_id":              video_data["video_id"],
        "channel_id":            channel_data["channel_id"],
        "channel_name":          channel_data.get("channel_name", ""),
        "title":                 video_data.get("title", ""),
        "published_at":          pub_at_str,
        "days_since_published":  days_since,
        "view_count":            int(video_data.get("view_count") or 0),
        "viral_score":           round(viral_score, 4),
        "prediction":            "CO KHA NANG VIRAL" if is_viral else "KHONG VIRAL",
        "confidence":            confidence,
        "top_5_drivers":         top_5_drivers,
        "recommendation":        recommendation,
        "warnings":              all_warnings,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 3 — FORMAT OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

def format_prediction_output(result: dict) -> str:
    """
    Render the prediction result dict as a human-readable terminal string.

    Parameters
    ----------
    result : dict
        Output of ``predict_from_api_data()``.

    Returns
    -------
    str
        Formatted text block.
    """
    lines: list[str] = []
    sep  = "=" * 60
    dash = "-" * 60

    lines.append("")
    lines.append(sep)
    lines.append(" KET QUA DU DOAN VIRAL")
    lines.append(sep)
    lines.append(f"  Kenh         : {result['channel_name']}")
    lines.append(f"  Video        : {result['title']}")
    lines.append(f"  Dang luc     : {result['published_at']}  ({result['days_since_published']} ngay truoc)")
    lines.append(f"  Luot xem     : {result['view_count']:,}")
    lines.append(dash)
    lines.append(f"  Viral Score  : {result['viral_score']:.2f}")
    lines.append(f"  Du doan      : {result['prediction']}")
    lines.append(f"  Do tu tin    : {result['confidence']}")
    lines.append(dash)
    lines.append("  TOP 5 YEU TO ANH HUONG:")
    for i, driver in enumerate(result.get("top_5_drivers", []), start=1):
        feat    = driver["feature"]
        impact  = driver["impact"]
        meaning = driver["meaning"]
        lines.append(f"  {i}. {feat:<30} {impact:+.4f}   {meaning}")
    lines.append(dash)
    lines.append(f"  DE XUAT: {result['recommendation']}")
    lines.append(sep)

    if result.get("warnings"):
        for w in result["warnings"]:
            lines.append(f"  [CANH BAO] {w}")

    lines.append("")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 4 — CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mock_video_data: dict = {
        "video_id":         "test_video_001",
        "title":            "10 Thu Thuat Python Ban Chua Biet",
        "description":      "Trong video nay toi se chia se 10 thu thuat Python...",
        "tags":             ["python", "lap trinh", "thu thuat"],
        "category_id":      "28",
        "default_language": "vi",
        "published_at":     "2024-11-10T18:00:00Z",
        "duration_iso8601": "PT12M30S",
        "has_caption":      True,
        "definition":       "hd",
        "is_embeddable":    True,
        "is_made_for_kids": False,
        "view_count":       45000,
        "like_count":       2100,
        "comment_count":    180,
    }

    mock_channel_data: dict = {
        "channel_id":        "UC_test_channel",
        "channel_name":      "Test Channel",
        "subscriber_count":  120000,
        "total_video_count": 85,
        "channel_created_at": "2020-03-15T00:00:00Z",
        "country_code":      "VN",
    }

    print("\n>>> Chay prediction voi mock data …")
    result = predict_from_api_data(mock_video_data, mock_channel_data)
    print(format_prediction_output(result))
