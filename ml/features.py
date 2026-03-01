"""
Feature engineering module for YouTube viral video prediction.

Transforms raw intermediate-layer data into a model-ready feature matrix.
This module is designed to work identically during both training and prediction.

Pipeline order:
    data_loader.py  →  label.py  →  features.py  →  train.py

Usage (standalone):1
    python -m ml.features
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Canonical feature list (training order reference) ────────────────────────
FEATURE_COLS: list[str] = [
    # Nhóm A — Temporal
    "published_hour", "published_dayofweek", "published_month",
    "is_weekend", "is_prime_time", "is_morning", "is_lunch_slot",
    "publish_quarter",
    # Nhóm B — Title
    "title_length", "title_word_count", "has_number", "has_question",
    "has_exclamation", "has_emoji", "has_caps_word",
    # Nhóm C — Content
    "duration_minutes", "tag_count", "has_tags", "is_hd",
    "has_caption", "is_embeddable", "is_made_for_kids",
    "is_shorts", "is_short", "is_medium", "category_id_enc",
    # Nhóm D — Channel
    "subscriber_log", "channel_age_days", "upload_freq_per_day",
    "channel_avg_views_log", "channel_avg_like_rate",
    "channel_avg_cmt_rate", "duration_vs_channel", "total_videos_crawled",
    # Nhóm E — Engagement
    "avg_views_per_day_log", "engagement_level_enc",
    "like_rate_pct", "comment_rate_pct", "engagement_score",
    # Nhóm F — Interaction
    "view_vs_channel_avg", "like_rate_vs_channel",
    "comment_rate_vs_channel", "velocity_score",
]

# Binary/boolean features — fill with 0 instead of median
_BINARY_FEATURES: set[str] = {
    "is_weekend", "is_prime_time", "is_morning", "is_lunch_slot",
    "has_number", "has_question", "has_exclamation", "has_emoji", "has_caps_word",
    "has_tags", "is_hd", "has_caption", "is_embeddable", "is_made_for_kids",
    "is_shorts", "is_short", "is_medium",
    "engagement_level_enc",
}


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 1 — MERGE DỮ LIỆU
# ═════════════════════════════════════════════════════════════════════════════

def merge_all_sources(
    df_labeled: pd.DataFrame,
    df_videos: pd.DataFrame,
    df_engagement: pd.DataFrame,
    df_channels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the four DataFrames into one wide DataFrame ready for feature engineering.

    Merge order:
        df_labeled  (base)
        ← LEFT JOIN df_videos       ON video_id
        ← LEFT JOIN df_engagement   ON video_id
        ← LEFT JOIN df_channels     ON channel_id

    Duplicate columns created by multiple joins are resolved by keeping the
    version from the highest-priority source (earlier in the join order).

    Parameters
    ----------
    df_labeled : pd.DataFrame
        Output of define_viral_label() — contains video_id, channel_id, is_viral.
    df_videos : pd.DataFrame
        Output of load_videos_enhanced() — int_videos__enhanced.
    df_engagement : pd.DataFrame
        Output of load_engagement_metrics() — int_engagement_metrics.
    df_channels : pd.DataFrame
        Output of load_channel_summary() — int_channel_summary.

    Returns
    -------
    pd.DataFrame
        One row per video with all source columns merged.

    Raises
    ------
    ValueError
        If any source DataFrame is empty, or if the result is empty after merge.
    """
    for name, df in [("df_labeled", df_labeled), ("df_videos", df_videos),
                     ("df_engagement", df_engagement), ("df_channels", df_channels)]:
        if df.empty:
            raise ValueError(f"{name} is empty — cannot merge sources.")

    n_before = len(df_labeled)
    logger.info("Bắt đầu merge: %d videos từ df_labeled", n_before)

    # 1. Base: labeled videos
    df = df_labeled[["video_id", "channel_id", "is_viral",
                     "view_ratio", "velocity_score", "days_since_published",
                     "label_strategy"]].copy()

    # 2. LEFT JOIN df_videos ON video_id
    df = df.merge(df_videos, on="video_id", how="left", suffixes=("", "_vid"))
    _drop_duplicate_cols(df, suffix="_vid")

    # 3. LEFT JOIN df_engagement ON video_id
    df = df.merge(df_engagement, on="video_id", how="left", suffixes=("", "_eng"))
    _drop_duplicate_cols(df, suffix="_eng")

    # 4. LEFT JOIN df_channels ON channel_id
    df = df.merge(
        df_channels.add_suffix("_ch").rename(columns={"channel_id_ch": "channel_id"}),
        on="channel_id",
        how="left",
    )
    # Resolve any remaining _ch columns: if no base column exists rename, else drop
    for col in list(df.columns):
        if col.endswith("_ch"):
            base = col[:-3]
            if base not in df.columns:
                df.rename(columns={col: base}, inplace=True)
            else:
                df.drop(columns=[col], inplace=True)

    n_after = len(df)
    if n_after == 0:
        raise ValueError("DataFrame is empty after merge — check source data.")
    if n_after != n_before:
        logger.warning(
            "Merge thay đổi số dòng: %d → %d  (delta %+d)",
            n_before, n_after, n_after - n_before,
        )
    else:
        logger.info("Merge hoàn tất: %d rows × %d cols", n_after, df.shape[1])

    return df


def _drop_duplicate_cols(df: pd.DataFrame, suffix: str) -> None:
    """Drop suffixed duplicate columns in-place, keeping the original."""
    cols_to_drop = [c for c in df.columns if c.endswith(suffix)]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 2 — HÀM FEATURE THEO TỪNG NHÓM (private)
# ═════════════════════════════════════════════════════════════════════════════

# ── Nhóm A: Temporal ─────────────────────────────────────────────────────────

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features derived from published_hour / dayofweek / month."""
    df = df.copy()

    hour      = pd.to_numeric(df.get("published_hour",      0), errors="coerce").fillna(0)
    dow       = pd.to_numeric(df.get("published_dayofweek", 0), errors="coerce").fillna(0)
    month     = pd.to_numeric(df.get("published_month",     1), errors="coerce").fillna(1)

    # BigQuery: dayofweek 1 = Sunday, 7 = Saturday
    df["is_weekend"]    = dow.isin([1, 7]).astype(int)
    df["is_prime_time"] = hour.between(18, 22).astype(int)
    df["is_morning"]    = hour.between(6, 10).astype(int)
    df["is_lunch_slot"] = hour.between(11, 13).astype(int)
    df["publish_quarter"] = ((month - 1) // 3 + 1).astype(int)

    # Ensure originals are present as numeric
    df["published_hour"]      = hour.astype(int)
    df["published_dayofweek"] = dow.astype(int)
    df["published_month"]     = month.astype(int)

    return df


# ── Nhóm B: Title ────────────────────────────────────────────────────────────

_RE_NUMBER    = re.compile(r"\d")
_RE_EMOJI     = re.compile(r"[\U0001F300-\U0001FFFF]")
_RE_CAPS_WORD = re.compile(r"\b[A-Z]{3,}\b")


def _add_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add NLP-lite features extracted from the video title string."""
    df = df.copy()

    titles = df.get("title", pd.Series([""] * len(df), index=df.index)).fillna("")

    df["title_length"]     = titles.str.len()
    df["title_word_count"] = titles.apply(lambda t: len(t.split()) if t else 0)
    df["has_number"]       = titles.apply(lambda t: int(bool(_RE_NUMBER.search(t))))
    df["has_question"]     = titles.str.contains("?", regex=False).astype(int)
    df["has_exclamation"]  = titles.str.contains("!", regex=False).astype(int)
    df["has_emoji"]        = titles.apply(lambda t: int(bool(_RE_EMOJI.search(t))))
    df["has_caps_word"]    = titles.apply(lambda t: int(bool(_RE_CAPS_WORD.search(t))))

    return df


# ── Nhóm C: Content ──────────────────────────────────────────────────────────

def _add_content_features(df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    """
    Add content-based features including duration, tags, definition, and
    encoded categorical fields.

    Parameters
    ----------
    label_encoders : dict
        Mutable dict.  If "category_id" key exists → transform (predict mode).
        Otherwise → fit_transform and store (train mode).
    """
    df = df.copy()

    # Duration
    dur_secs = pd.to_numeric(df.get("duration_seconds", 0), errors="coerce").fillna(0)
    df["duration_minutes"] = dur_secs / 60

    # Tags
    tags = df.get("tags", pd.Series([None] * len(df), index=df.index))
    df["tag_count"] = tags.apply(
        lambda t: len(str(t).split(",")) if pd.notna(t) and str(t).strip() else 0
    )
    df["has_tags"] = (df["tag_count"] > 0).astype(int)

    # Definition
    definition = df.get("definition", pd.Series([""] * len(df), index=df.index)).fillna("")
    df["is_hd"] = (definition.str.lower() == "hd").astype(int)

    # Boolean flags
    for col in ("has_caption", "is_embeddable", "is_made_for_kids"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    # One-hot encode video_length_category (drop is_long as reference)
    vlc = df.get("video_length_category",
                 pd.Series(["short"] * len(df), index=df.index)).fillna("short")
    df["is_shorts"] = (vlc == "shorts").astype(int)
    df["is_short"]  = (vlc == "short").astype(int)
    df["is_medium"] = (vlc == "medium").astype(int)
    # is_long intentionally omitted → reference category

    # Label encode category_id
    cat_raw = df.get("category_id",
                     pd.Series(["unknown"] * len(df), index=df.index)).fillna("unknown").astype(str)

    if "category_id" in label_encoders:
        le: LabelEncoder = label_encoders["category_id"]
        known_classes = set(le.classes_)
        cat_mapped = cat_raw.apply(lambda c: c if c in known_classes else "__unknown__")
        # Ensure __unknown__ is in classes
        if "__unknown__" not in known_classes:
            le.classes_ = np.append(le.classes_, "__unknown__")
        df["category_id_enc"] = le.transform(cat_mapped)
    else:
        le = LabelEncoder()
        df["category_id_enc"] = le.fit_transform(cat_raw)
        label_encoders["category_id"] = le
        logger.info("LabelEncoder for category_id fitted with %d classes", len(le.classes_))

    return df


# ── Nhóm D: Channel ──────────────────────────────────────────────────────────

def _add_channel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add channel-level context features."""
    df = df.copy()

    subs = pd.to_numeric(df.get("subscriber_count", df.get("channel_subscribers", 0)),
                         errors="coerce").fillna(0).clip(lower=0)
    df["subscriber_log"] = np.log1p(subs)

    # Channel age days  (published_at − channel_created_at)
    pub_at  = pd.to_datetime(df.get("published_at"),  utc=True, errors="coerce")
    chan_at = pd.to_datetime(df.get("channel_created_at"), utc=True, errors="coerce")
    if pub_at.notna().any() and chan_at.notna().any():
        age = (pub_at - chan_at).dt.days.clip(lower=0)
    else:
        age = pd.Series(0, index=df.index)
    df["channel_age_days"] = age.fillna(0).astype(float)

    # Upload frequency
    avg_days_between = pd.to_numeric(
        df.get("avg_days_between_uploads", None), errors="coerce"
    ).fillna(0)
    df["upload_freq_per_day"] = np.where(avg_days_between > 0, 1 / avg_days_between, 0.0)

    # Channel avg views (log)
    avg_views = pd.to_numeric(df.get("avg_views_per_video", 0), errors="coerce").fillna(0)
    df["channel_avg_views_log"] = np.log1p(avg_views)

    # Channel engagement rates
    df["channel_avg_like_rate"] = pd.to_numeric(
        df.get("avg_like_rate_pct", 0), errors="coerce"
    ).fillna(0)
    df["channel_avg_cmt_rate"] = pd.to_numeric(
        df.get("avg_comment_rate_pct", 0), errors="coerce"
    ).fillna(0)

    # Duration vs channel average
    dur_secs  = pd.to_numeric(df.get("duration_seconds", 0), errors="coerce").fillna(0)
    avg_dur   = pd.to_numeric(df.get("avg_video_duration_seconds", 1), errors="coerce").fillna(1).clip(lower=1)
    df["duration_vs_channel"] = (dur_secs / avg_dur).fillna(1)

    # Total videos crawled
    df["total_videos_crawled"] = pd.to_numeric(
        df.get("total_videos_crawled", 0), errors="coerce"
    ).fillna(0)

    return df


# ── Nhóm E: Engagement ───────────────────────────────────────────────────────

def _add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-video engagement features."""
    df = df.copy()

    avg_vpd = pd.to_numeric(df.get("avg_views_per_day", 0), errors="coerce").fillna(0)
    df["avg_views_per_day_log"] = np.log1p(avg_vpd)

    level_map = {"high": 2, "medium": 1, "low": 0}
    df["engagement_level_enc"] = (
        df.get("engagement_level",
               pd.Series(["low"] * len(df), index=df.index))
        .fillna("low")
        .map(level_map)
        .fillna(0)
        .astype(int)
    )

    for col in ("like_rate_pct", "comment_rate_pct", "engagement_score"):
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ── Nhóm F: Interaction ──────────────────────────────────────────────────────

def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-source interaction / ratio features."""
    df = df.copy()

    view_count  = pd.to_numeric(df.get("view_count", 0), errors="coerce").fillna(0)
    avg_views   = pd.to_numeric(df.get("avg_views_per_video", 1), errors="coerce").fillna(1).clip(lower=1)
    like_rate   = pd.to_numeric(df.get("like_rate_pct", 0), errors="coerce").fillna(0)
    ch_like     = pd.to_numeric(df.get("avg_like_rate_pct", 1), errors="coerce").fillna(1).clip(lower=1e-9)
    cmt_rate    = pd.to_numeric(df.get("comment_rate_pct", 0), errors="coerce").fillna(0)
    ch_cmt      = pd.to_numeric(df.get("avg_comment_rate_pct", 1), errors="coerce").fillna(1).clip(lower=1e-9)
    avg_vpd     = pd.to_numeric(df.get("avg_views_per_day", 0), errors="coerce").fillna(0)
    subs        = pd.to_numeric(
        df.get("subscriber_count", df.get("channel_subscribers", 1)),
        errors="coerce",
    ).fillna(1).clip(lower=1)

    df["view_vs_channel_avg"]     = (view_count / avg_views).fillna(0).clip(upper=100)
    df["like_rate_vs_channel"]    = (like_rate  / ch_like).fillna(1).clip(upper=10)
    df["comment_rate_vs_channel"] = (cmt_rate   / ch_cmt).fillna(1).clip(upper=10)
    df["velocity_score"]          = (avg_vpd / (subs * 0.01)).fillna(0).clip(upper=1000)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 3 — HÀM FEATURE ENGINEERING CHÍNH
# ═════════════════════════════════════════════════════════════════════════════

def engineer_features(
    df: pd.DataFrame,
    label_encoders: Optional[dict] = None,
    feature_list: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run all feature engineering groups and return a model-ready feature matrix.

    This function is designed to be called identically during training and
    prediction — pass ``label_encoders`` and ``feature_list`` (saved from
    training) when predicting.

    Parameters
    ----------
    df : pd.DataFrame
        Output of merge_all_sources().
    label_encoders : dict, optional
        Dict of fitted sklearn encoders keyed by column name.
        Pass None on first training run (encoders will be fit in-place).
    feature_list : list of str, optional
        The ordered list of feature column names used during training.
        Pass None during training (FEATURE_COLS used).
        Pass the saved list during prediction to ensure column alignment.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix — no target column, no ID columns.
    label_encoders : dict
        Fitted encoders (reuse during prediction).

    Raises
    ------
    ValueError
        If df is empty after merge.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty — cannot engineer features.")

    if label_encoders is None:
        label_encoders = {}

    logger.info("Feature engineering: %d rows as input", len(df))

    # ── Apply all feature groups ──────────────────────────────────────────────
    df = _add_temporal_features(df)
    logger.info("  [A] Temporal features done")

    df = _add_title_features(df)
    logger.info("  [B] Title features done")

    df = _add_content_features(df, label_encoders)
    logger.info("  [C] Content features done")

    df = _add_channel_features(df)
    logger.info("  [D] Channel features done")

    df = _add_engagement_features(df)
    logger.info("  [E] Engagement features done")

    df = _add_interaction_features(df)
    logger.info("  [F] Interaction features done")

    # ── Select & align feature columns ───────────────────────────────────────
    cols_to_use = feature_list if feature_list is not None else FEATURE_COLS

    X = pd.DataFrame(index=df.index)
    for col in cols_to_use:
        if col in df.columns:
            X[col] = df[col]
        else:
            logger.warning("[CANH BAO] Cot '%s' khong ton tai — dien gia tri 0", col)
            X[col] = 0

    X = X[cols_to_use]  # enforce exact order

    # ── Quality report ────────────────────────────────────────────────────────
    n_features = X.shape[1]
    missing_rates = (X.isna().mean() * 100).round(2)
    high_missing = missing_rates[missing_rates > 5]

    print()
    print("=" * 60)
    print(" FEATURE ENGINEERING REPORT")
    print("=" * 60)
    print(f"  Tong features    : {n_features}")
    if high_missing.empty:
        print("  Missing rate > 5%: Khong co")
    else:
        print("  Missing rate > 5%:")
        for c, pct in high_missing.items():
            print(f"    {c:<35} : {pct:.1f}%")

    # High correlation check
    numeric_X = X.select_dtypes(include=[np.number])
    if numeric_X.shape[1] > 1:
        corr_matrix = numeric_X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr = [
            (c1, c2, corr_matrix.loc[c1, c2])
            for c1 in upper.index
            for c2 in upper.columns
            if pd.notna(upper.loc[c1, c2]) and upper.loc[c1, c2] > 0.90
        ]
        if high_corr:
            print("  Correlation > 0.90:")
            for c1, c2, v in high_corr[:10]:
                print(f"    {c1:<30} ↔ {c2:<30} : {v:.3f}")
        else:
            print("  Correlation > 0.90: Khong co")
    print("=" * 60)
    print()

    logger.info("engineer_features hoàn tất: X shape = %s", X.shape)
    return X, label_encoders


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 4 — HÀM FILL MISSING
# ═════════════════════════════════════════════════════════════════════════════

def fill_missing(
    X: pd.DataFrame,
    train_medians: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Fill NaN values in the feature matrix.

    Rules:
    - Binary / boolean features → fill with 0
    - ``engagement_level_enc`` → fill with 0 (represents "low")
    - All other numeric features → fill with median
      (computed from X if train_medians is None, else reuse saved medians)

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix from engineer_features().
    train_medians : dict, optional
        Dict mapping column name → training-time median.
        Pass None during training (medians computed from X).
        Pass the saved dict during prediction to avoid leakage.

    Returns
    -------
    X_filled : pd.DataFrame
    train_medians : dict
        Computed or passed-through medians (save this for prediction).
    """
    X = X.copy()
    is_training = train_medians is None
    if is_training:
        train_medians = {}

    filled_report = []

    for col in X.columns:
        n_missing = int(X[col].isna().sum())
        if n_missing == 0:
            continue

        pct_missing = n_missing / len(X) * 100

        if col in _BINARY_FEATURES or col == "engagement_level_enc":
            fill_val = 0
            X[col] = X[col].fillna(fill_val)
            fill_type = "0 (binary)"

        else:
            if is_training:
                median_val = X[col].median()
                if pd.isna(median_val):
                    logger.warning(
                        "Cot '%s' toan bo la NaN — fill = 0", col
                    )
                    median_val = 0.0
                train_medians[col] = float(median_val)
            else:
                median_val = train_medians.get(col, 0.0)

            X[col] = X[col].fillna(median_val)
            fill_val = median_val
            fill_type = f"median = {median_val:.4f}"

        filled_report.append((col, pct_missing, fill_type))

    if filled_report:
        print()
        print("=" * 60)
        print(" FILL MISSING VALUES")
        print("=" * 60)
        print(f"  {'Cot':<35} {'Missing%':>9}  Fill")
        print("-" * 60)
        for col, pct, fill_type in filled_report:
            print(f"  {col:<35} {pct:>8.1f}%  {fill_type}")
        print("=" * 60)
        print()
    else:
        logger.info("fill_missing: Không có cột nào thiếu dữ liệu")

    return X, train_medians


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 5 — CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from ml.data_loader import load_all_intermediate_data
    from ml.label import define_viral_label

    # 1. Load dữ liệu
    print("\n>>> BƯỚC 1: Load dữ liệu từ BigQuery …")
    data = load_all_intermediate_data()

    # 2. Định nghĩa nhãn
    print("\n>>> BƯỚC 2: Định nghĩa nhãn viral …")
    df_labeled = define_viral_label(
        df_engagement=data["engagement"],
        df_channels=data["channels"],
        strategy="auto",
    )

    # 3. Merge tất cả nguồn
    print("\n>>> BƯỚC 3: Merge all sources …")
    df_merged = merge_all_sources(
        df_labeled=df_labeled,
        df_videos=data["videos"],
        df_engagement=data["engagement"],
        df_channels=data["channels"],
    )
    print(f"    df_merged shape: {df_merged.shape}")

    # 4. Feature engineering
    print("\n>>> BƯỚC 4: Feature engineering …")
    X, label_encoders = engineer_features(df_merged)

    # 5. Fill missing
    print("\n>>> BƯỚC 5: Fill missing values …")
    X, train_medians = fill_missing(X)

    # 6. Target
    y = df_merged["is_viral"]

    # 7. Kết quả
    print("\n>>> KẾT QUẢ:")
    print(f"    X shape          : {X.shape}")
    print(f"    y distribution   : {dict(y.value_counts().sort_index())}")
    print(f"    Label encoders   : {list(label_encoders.keys())}")
    print(f"    Saved medians    : {len(train_medians)} cột")

    print("\n>>> Danh sách features:")
    for i, col in enumerate(X.columns, 1):
        print(f"    {i:>2}. {col}")

    print("\n>>> 5 dòng đầu:")
    print(X.head(5).to_string())
