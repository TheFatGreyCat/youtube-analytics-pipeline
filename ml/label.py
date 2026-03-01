"""
Label definition module for YouTube viral video prediction.

Defines the binary is_viral label (0/1) for each video, based on multiple
strategies derived from engagement metrics and channel-level baselines.

Pipeline order:
    data_loader.py  →  label.py  →  features.py  →  train.py

Usage (standalone):1
    python -m ml.label
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Logger (consistent with data_loader.py) ──────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 1 — PHÂN TÍCH DỮ LIỆU TRƯỚC KHI ĐỊNH NGHĨA NHÃN
# ═════════════════════════════════════════════════════════════════════════════

def analyze_label_candidates(
    df_engagement: pd.DataFrame,
    df_channels: pd.DataFrame,
) -> None:
    """
    Run EDA to understand data distribution BEFORE deciding the viral label.

    Prints a detailed report and renders matplotlib charts.
    Does not return anything — purely diagnostic.

    Parameters
    ----------
    df_engagement : pd.DataFrame
        Output of load_engagement_metrics() — one row per video.
    df_channels : pd.DataFrame
        Output of load_channel_summary() — one row per channel.

    Raises
    ------
    ValueError
        If df_engagement is empty.
    KeyError
        If required columns are missing from df_engagement.
    """
    if df_engagement.empty:
        raise ValueError("df_engagement is empty — cannot analyse label candidates.")

    required_cols = {"view_count", "channel_id", "avg_views_per_day", "channel_subscribers"}
    missing = required_cols - set(df_engagement.columns)
    if missing:
        raise KeyError(
            f"Missing columns in df_engagement: {missing}. "
            "Check that data_loader.py returned int_engagement_metrics correctly."
        )

    print()
    print("=" * 60)
    print(" EDA — PHÂN TÍCH ỨNG VIÊN NHÃN VIRAL")
    print("=" * 60)

    # ── BƯỚC 1 — is_potentially_viral có sẵn ─────────────────────────────────
    print("\n[BƯỚC 1] is_potentially_viral (nhãn có sẵn)")
    print("-" * 50)
    if "is_potentially_viral" in df_engagement.columns:
        vc = df_engagement["is_potentially_viral"].value_counts()
        n_true  = int(vc.get(True,  0))
        n_false = int(vc.get(False, 0))
        total   = n_true + n_false
        viral_rate = n_true / total * 100 if total > 0 else 0.0

        print(f"  True  : {n_true:,}  ({viral_rate:.1f}%)")
        print(f"  False : {n_false:,}  ({100 - viral_rate:.1f}%)")

        if 8.0 <= viral_rate <= 20.0:
            print(f"  → Danh gia: Co the dung duoc  (viral_rate={viral_rate:.1f}% nam trong 8-20%)")
        else:
            print(f"  → Danh gia: Can dinh nghia lai  (viral_rate={viral_rate:.1f}% nam NGOAI 8-20%)")
    else:
        print("  [!] Cot is_potentially_viral khong ton tai trong df_engagement.")
        viral_rate = 0.0

    # ── BƯỚC 2 — Phân phối view_count ────────────────────────────────────────
    print("\n[BƯỚC 2] Phân phối view_count")
    print("-" * 50)
    vc_series = df_engagement["view_count"].dropna()
    pcts = [10, 25, 50, 75, 90, 95, 99]
    quantile_vals = np.percentile(vc_series, pcts)
    print(f"  Min    : {vc_series.min():>12,.0f}")
    for p, v in zip(pcts, quantile_vals):
        label = f"p{p:02d}" if p != 50 else "Median"
        print(f"  {label:<6} : {v:>12,.0f}")
    print(f"  Max    : {vc_series.max():>12,.0f}")
    print(f"  Mean   : {vc_series.mean():>12,.0f}")
    print(f"  Std    : {vc_series.std():>12,.0f}")

    # Histogram of log1p(view_count)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(np.log1p(vc_series), bins=60, color="#4C72B0", alpha=0.75, edgecolor="white")
    for p, col in zip([75, 90, 95], ["#DD8452", "#C44E52", "#8172B2"]):
        val = np.percentile(vc_series, p)
        ax.axvline(np.log1p(val), color=col, linestyle="--", linewidth=1.6, label=f"p{p} = {val:,.0f}")
    ax.set_xlabel("log1p(view_count)")
    ax.set_ylabel("Số video")
    ax.set_title("Phân phối log1p(view_count)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ── BƯỚC 3 — Phân phối view_ratio ────────────────────────────────────────
    print("\n[BƯỚC 3] Phân phối view_ratio (video / trung bình kênh)")
    print("-" * 50)
    merged = df_engagement.merge(
        df_channels[["channel_id", "avg_views_per_video"]],
        on="channel_id",
        how="left",
    )
    avg_col = merged["avg_views_per_video"].fillna(1).clip(lower=1)
    merged["view_ratio"] = merged["view_count"] / avg_col

    for p in [50, 75, 90, 95, 99]:
        v = np.percentile(merged["view_ratio"].dropna(), p)
        label = "Median" if p == 50 else f"p{p:02d}"
        print(f"  {label:<6} : {v:.2f}x")

    for threshold in [2.0, 3.0, 5.0]:
        pct_above = (merged["view_ratio"] >= threshold).mean() * 100
        print(f"  view_ratio >= {threshold:.0f}x : {pct_above:.1f}% video")

    # ── BƯỚC 4 — Phân phối velocity_score ────────────────────────────────────
    print("\n[BƯỚC 4] Phân phối velocity_score (tốc độ view / subscriber)")
    print("-" * 50)
    subs = df_engagement["channel_subscribers"].fillna(1).clip(lower=1)
    velocity = df_engagement["avg_views_per_day"] / (subs * 0.01)
    velocity = velocity.replace([np.inf, -np.inf], np.nan).dropna()

    for p in [50, 75, 90, 95]:
        v = np.percentile(velocity, p)
        label = "Median" if p == 50 else f"p{p:02d}"
        print(f"  {label:<6} : {v:.3f}")
    pct_v1 = (velocity >= 1.0).mean() * 100
    print(f"  velocity >= 1.0 : {pct_v1:.1f}% video")

    # ── BƯỚC 5 — Gợi ý threshold ─────────────────────────────────────────────
    ratio_3x_rate = (merged["view_ratio"] >= 3.0).mean() * 100
    combined_rate = (
        (merged["view_ratio"] >= 3.0) | (velocity.reindex(merged.index) >= 1.0)
    ).mean() * 100

    existing_eval = (
        "Co the dung duoc" if 8.0 <= viral_rate <= 20.0 else "Can xem xet lai"
    )

    print()
    print("=" * 60)
    print(" GỢI Ý THRESHOLD CHO NHÃN VIRAL")
    print("=" * 60)
    print(f"  Phuong an A — Dung is_potentially_viral co san:")
    print(f"    Viral rate hien tai: {viral_rate:.1f}%")
    print(f"    Danh gia: {existing_eval}")
    print()
    print(f"  Phuong an B — Dua tren view_ratio >= 3x trung binh kenh:")
    print(f"    Uoc tinh viral rate: {ratio_3x_rate:.1f}%")
    print()
    print(f"  Phuong an C — Ket hop view_ratio >= 3x VA/HOAC velocity >= 1.0:")
    print(f"    Uoc tinh viral rate: {combined_rate:.1f}%")
    print()

    # Recommend
    candidates = {
        "A (existing)": abs(viral_rate  - 14),
        "B (ratio)":    abs(ratio_3x_rate - 14),
        "C (combined)": abs(combined_rate  - 14),
    }
    best = min(candidates, key=candidates.get)
    print(f"  Khuyen nghi: {best} — gan viral_rate nhat muc ly tuong 8-20%")
    print("=" * 60)
    print()


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 2 — HÀM ĐỊNH NGHĨA NHÃN CHÍNH
# ═════════════════════════════════════════════════════════════════════════════

def define_viral_label(
    df_engagement: pd.DataFrame,
    df_channels: pd.DataFrame,
    strategy: str = "auto",
    view_ratio_threshold: float = 3.0,
    velocity_threshold: float = 1.0,
    min_days_published: int = 7,
) -> pd.DataFrame:
    """
    Assign binary viral label (is_viral = 0 or 1) to each video.

    Parameters
    ----------
    df_engagement : pd.DataFrame
        Output of load_engagement_metrics() — one row per video.
    df_channels : pd.DataFrame
        Output of load_channel_summary() — one row per channel.
    strategy : str, optional
        Label strategy:
          "auto"     — auto-select based on existing viral_rate
          "existing" — use is_potentially_viral column as-is
          "ratio"    — view_ratio >= view_ratio_threshold
          "combined" — view_ratio >= threshold OR velocity >= threshold
        Default: "auto".
    view_ratio_threshold : float, optional
        Multiplier above channel avg_views_per_video to be viral. Default: 3.0.
    velocity_threshold : float, optional
        Minimum velocity_score (avg_views_per_day / subs*0.01). Default: 1.0.
    min_days_published : int, optional
        Exclude videos published fewer than N days ago. Default: 7.

    Returns
    -------
    pd.DataFrame
        Columns: video_id, channel_id, view_count, days_since_published,
                 view_ratio, velocity_score, is_viral (int), label_strategy (str).

    Raises
    ------
    ValueError
        If df_engagement is empty or strategy is unrecognised.
    KeyError
        If required columns are missing.
    """
    if df_engagement.empty:
        raise ValueError("df_engagement is empty — cannot define viral label.")

    valid_strategies = {"auto", "existing", "ratio", "combined"}
    if strategy not in valid_strategies:
        raise ValueError(f"strategy='{strategy}' is invalid. Choose from {valid_strategies}.")

    required_eng = {"video_id", "channel_id", "view_count", "days_since_published",
                    "avg_views_per_day", "channel_subscribers"}
    missing_eng = required_eng - set(df_engagement.columns)
    if missing_eng:
        raise KeyError(
            f"Missing columns in df_engagement: {missing_eng}. "
            "Check that data_loader.py returned int_engagement_metrics correctly."
        )

    required_ch = {"channel_id", "avg_views_per_video"}
    missing_ch = required_ch - set(df_channels.columns)
    if missing_ch:
        raise KeyError(
            f"Missing columns in df_channels: {missing_ch}. "
            "Check that data_loader.py returned int_channel_summary correctly."
        )

    df = df_engagement.copy()

    # ── BƯỚC 1 — Lọc video đủ điều kiện ─────────────────────────────────────
    n_before = len(df)
    df = df[df["days_since_published"] >= min_days_published].copy()
    n_after  = len(df)
    n_removed = n_before - n_after
    logger.info(
        "Loc video: loai %d video published < %d ngay | con lai %d video",
        n_removed, min_days_published, n_after,
    )

    if n_after == 0:
        raise ValueError(
            f"Sau khi loc min_days_published >= {min_days_published}, "
            "khong con video nao. Giam gia tri min_days_published."
        )

    # ── BƯỚC 2 — JOIN với df_channels ────────────────────────────────────────
    channel_cols = df_channels[["channel_id", "avg_views_per_video"]].copy()
    df = df.merge(channel_cols, on="channel_id", how="left", suffixes=("", "_ch"))

    n_unmatched = df["avg_views_per_video"].isna().sum()
    if n_unmatched > 0:
        logger.warning(
            "%d video khong join duoc voi df_channels — avg_views_per_video se duoc set = 1",
            n_unmatched,
        )
    df["avg_views_per_video"] = df["avg_views_per_video"].fillna(1).clip(lower=1)

    # ── BƯỚC 3 — Tính view_ratio và velocity_score ───────────────────────────
    df["view_ratio"] = df["view_count"] / df["avg_views_per_video"]

    subs = df["channel_subscribers"].fillna(1).clip(lower=1)
    df["velocity_score"] = df["avg_views_per_day"] / (subs * 0.01)
    df["velocity_score"] = df["velocity_score"].replace([np.inf, -np.inf], 0).fillna(0)

    # ── BƯỚC 4 — Chọn strategy và tạo nhãn ──────────────────────────────────
    resolved_strategy = strategy

    if strategy == "auto":
        if "is_potentially_viral" in df.columns:
            existing_rate = df["is_potentially_viral"].mean() * 100
        else:
            existing_rate = 0.0

        if 8.0 <= existing_rate <= 20.0:
            resolved_strategy = "existing"
            logger.info(
                "auto → 'existing'  (is_potentially_viral rate = %.1f%%, trong 8-20%%)",
                existing_rate,
            )
        else:
            resolved_strategy = "combined"
            logger.info(
                "auto → 'combined'  (is_potentially_viral rate = %.1f%%, ngoai 8-20%%)",
                existing_rate,
            )

    def _apply_strategy(df_: pd.DataFrame, strat: str, vr_thr: float, vel_thr: float) -> pd.Series:
        if strat == "existing":
            if "is_potentially_viral" not in df_.columns:
                raise KeyError(
                    "strategy='existing' nhung khong co cot is_potentially_viral. "
                    "Thu strategy='ratio' hoac 'combined'."
                )
            return df_["is_potentially_viral"].astype(int)
        elif strat == "ratio":
            return (df_["view_ratio"] >= vr_thr).astype(int)
        elif strat == "combined":
            return (
                (df_["view_ratio"] >= vr_thr) | (df_["velocity_score"] >= vel_thr)
            ).astype(int)
        else:
            raise ValueError(f"Unknown resolved strategy: {strat}")

    df["is_viral"] = _apply_strategy(df, resolved_strategy, view_ratio_threshold, velocity_threshold)

    # ── BƯỚC 5 — Auto-adjust threshold nếu viral_rate ngoài [5%, 30%] ────────
    current_vr_thr = view_ratio_threshold
    MAX_ITER = 5

    for iteration in range(MAX_ITER):
        vr = df["is_viral"].mean() * 100

        if vr < 5.0 and resolved_strategy in ("ratio", "combined"):
            new_thr = round(current_vr_thr - 0.5, 1)
            logger.warning(
                "[CANH BAO] Viral rate %.1f%% qua thap. Da tu dong dieu chinh threshold xuong %.1fx  (iteration %d)",
                vr, new_thr, iteration + 1,
            )
            current_vr_thr = new_thr
            df["is_viral"] = _apply_strategy(df, resolved_strategy, current_vr_thr, velocity_threshold)

        elif vr > 30.0 and resolved_strategy in ("ratio", "combined"):
            new_thr = round(current_vr_thr + 0.5, 1)
            logger.warning(
                "[CANH BAO] Viral rate %.1f%% qua cao. Da tu dong dieu chinh threshold len %.1fx  (iteration %d)",
                vr, new_thr, iteration + 1,
            )
            current_vr_thr = new_thr
            df["is_viral"] = _apply_strategy(df, resolved_strategy, current_vr_thr, velocity_threshold)

        else:
            break
    else:
        final_vr = df["is_viral"].mean() * 100
        logger.warning(
            "[CANH BAO] Sau %d lan thu, viral rate van la %.1f%%. Tiep tuc voi threshold %.1fx.",
            MAX_ITER, final_vr, current_vr_thr,
        )

    # ── BƯỚC 6 — In báo cáo ─────────────────────────────────────────────────
    total       = len(df)
    n_viral     = int(df["is_viral"].sum())
    n_not_viral = total - n_viral
    final_rate  = n_viral / total * 100 if total > 0 else 0.0

    viral_ratios     = df.loc[df["is_viral"] == 1, "view_ratio"]
    non_viral_ratios = df.loc[df["is_viral"] == 0, "view_ratio"]

    print()
    print("=" * 60)
    print(" KET QUA DINH NGHIA NHAN VIRAL")
    print("=" * 60)
    print(f"  Strategy          : {resolved_strategy}")
    print(f"  Threshold dung    : view_ratio >= {current_vr_thr:.1f}x  HOAC  velocity >= {velocity_threshold:.1f}")
    print(f"  Min days published: {min_days_published} ngay")
    print("-" * 60)
    print(f"  Tong video sau loc       : {total:,}")
    print(f"  Video viral (is_viral=1) : {n_viral:,}  ({final_rate:.1f}%)")
    print(f"  Video khong viral        : {n_not_viral:,}  ({100 - final_rate:.1f}%)")
    print("-" * 60)
    if not viral_ratios.empty:
        print("  Phan phoi view_ratio cua video VIRAL:")
        print(f"    Median : {viral_ratios.median():.1f}x")
        print(f"    p75    : {viral_ratios.quantile(0.75):.1f}x")
        print(f"    Max    : {viral_ratios.max():.1f}x")
    if not non_viral_ratios.empty:
        print("  Phan phoi view_ratio cua video KHONG VIRAL:")
        print(f"    Median : {non_viral_ratios.median():.1f}x")
        print(f"    p75    : {non_viral_ratios.quantile(0.75):.1f}x")
        print(f"    Max    : {non_viral_ratios.max():.1f}x")
    print("=" * 60)
    print()

    # ── Build output DataFrame ───────────────────────────────────────────────
    out_cols = ["video_id", "channel_id", "view_count", "days_since_published",
                "view_ratio", "velocity_score", "is_viral"]
    df_out = df[out_cols].copy()
    df_out["is_viral"] = df_out["is_viral"].astype(int)
    df_out["label_strategy"] = resolved_strategy

    logger.info(
        "define_viral_label hoàn tất: %d video, %d viral (%.1f%%)",
        total, n_viral, final_rate,
    )
    return df_out


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 3 — KIỂM TRA CHẤT LƯỢNG NHÃN
# ═════════════════════════════════════════════════════════════════════════════

def validate_label_quality(
    df_labeled: pd.DataFrame,
    df_engagement: pd.DataFrame,
) -> None:
    """
    Validate the viral label for bias and potential data leakage.

    Parameters
    ----------
    df_labeled : pd.DataFrame
        Output of define_viral_label() — contains is_viral column.
    df_engagement : pd.DataFrame
        Original engagement metrics — used to JOIN published_at and
        video_length_category columns.

    Raises
    ------
    ValueError
        If df_labeled is empty or is_viral column is missing.
    """
    if df_labeled.empty:
        raise ValueError("df_labeled is empty — nothing to validate.")
    if "is_viral" not in df_labeled.columns:
        raise ValueError("df_labeled does not contain 'is_viral' column.")

    # Track pass/fail for summary
    checks: dict[str, str] = {}

    overall_rate = df_labeled["is_viral"].mean() * 100
    checks["Viral rate tong the"] = (
        f"PASS  → {overall_rate:.1f}%  (muc tieu 8-20%)"
        if 5.0 <= overall_rate <= 30.0
        else f"CANH BAO  → {overall_rate:.1f}%  (ngoai 5-30%)"
    )

    # ── KIỂM TRA 1 — Viral rate theo kênh ────────────────────────────────────
    print("\n[KIỂM TRA 1] Viral rate theo kênh")
    print("-" * 50)

    channel_stats = (
        df_labeled.groupby("channel_id")["is_viral"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "viral_videos", "count": "total_videos"})
    )
    channel_stats["viral_rate"] = channel_stats["viral_videos"] / channel_stats["total_videos"] * 100

    zero_rate_channels  = (channel_stats["viral_rate"] == 0.0).sum()
    full_rate_channels  = (channel_stats["viral_rate"] == 100.0).sum()

    if zero_rate_channels > 0:
        logger.warning(
            "%d kenh co viral_rate = 0%% — model se khong hoc duoc pattern cua cac kenh nay.",
            zero_rate_channels,
        )
    if full_rate_channels > 0:
        logger.warning(
            "%d kenh co viral_rate = 100%% — co the bi data leakage hoac threshold qua thap.",
            full_rate_channels,
        )

    checks["Kenh bi bias (100%)"] = (
        f"PASS  → {full_rate_channels} kenh"
        if full_rate_channels == 0
        else f"CANH BAO  → {full_rate_channels} kenh"
    )
    checks["Kenh bi bias (0%)"] = (
        f"PASS  → {zero_rate_channels} kenh"
        if zero_rate_channels == 0
        else f"CANH BAO  → {zero_rate_channels} kenh  (it du lieu)"
    )

    print(f"  Kenh viral_rate = 0%    : {zero_rate_channels}")
    print(f"  Kenh viral_rate = 100%  : {full_rate_channels}")

    top5_high = channel_stats.nlargest(5, "viral_rate")[["viral_rate", "total_videos"]]
    top5_low  = channel_stats.nsmallest(5, "viral_rate")[["viral_rate", "total_videos"]]

    print("\n  Top 5 kênh viral_rate CAO nhất:")
    for cid, row in top5_high.iterrows():
        print(f"    {cid[:30]:<30} → {row['viral_rate']:5.1f}%  ({int(row['total_videos'])} videos)")

    print("\n  Top 5 kênh viral_rate THẤP nhất:")
    for cid, row in top5_low.iterrows():
        print(f"    {cid[:30]:<30} → {row['viral_rate']:5.1f}%  ({int(row['total_videos'])} videos)")

    # ── KIỂM TRA 2 — Viral rate theo thời gian ───────────────────────────────
    print("\n[KIỂM TRA 2] Viral rate theo tháng")
    print("-" * 50)

    df_time = df_labeled.merge(
        df_engagement[["video_id", "published_at"]],
        on="video_id",
        how="left",
    )

    seasonal_ok = True
    if "published_at" in df_time.columns and df_time["published_at"].notna().any():
        df_time["published_at"] = pd.to_datetime(df_time["published_at"], utc=True, errors="coerce")
        df_time["year_month"] = df_time["published_at"].dt.to_period("M").astype(str)
        monthly = (
            df_time.groupby("year_month")["is_viral"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "viral_rate", "count": "n_videos"})
        )
        monthly["viral_rate_pct"] = monthly["viral_rate"] * 100

        print(f"  {'Thang':<10} {'Viral rate':>12} {'So video':>10}")
        for ym, row in monthly.iterrows():
            flag = ""
            if abs(row["viral_rate_pct"] - overall_rate) > 15:
                flag = "  ← CANH BAO"
                seasonal_ok = False
            print(f"  {ym:<10} {row['viral_rate_pct']:>11.1f}%  {int(row['n_videos']):>8}{flag}")
    else:
        print("  [!] Khong co cot published_at — bo qua kiem tra theo thoi gian.")

    checks["Seasonal trend"] = (
        "PASS  → On dinh" if seasonal_ok else "CANH BAO  → Co thang lech > 15%"
    )

    # ── KIỂM TRA 3 — Viral rate theo video_length_category ───────────────────
    print("\n[KIỂM TRA 3] Viral rate theo video_length_category")
    print("-" * 50)

    df_cat = df_labeled.merge(
        df_engagement[["video_id", "video_length_category"]],
        on="video_id",
        how="left",
    )
    if "video_length_category" in df_cat.columns and df_cat["video_length_category"].notna().any():
        cat_stats = (
            df_cat.groupby("video_length_category")["is_viral"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "viral_rate", "count": "n_videos"})
        )
        cat_stats["viral_rate_pct"] = cat_stats["viral_rate"] * 100

        shorts_rate = cat_stats.loc["shorts", "viral_rate_pct"] if "shorts" in cat_stats.index else None
        overall_non_shorts = cat_stats.loc[
            cat_stats.index != "shorts", "viral_rate_pct"
        ].mean() if len(cat_stats[cat_stats.index != "shorts"]) > 0 else None

        for cat, row in cat_stats.iterrows():
            print(f"  {cat:<10} → {row['viral_rate_pct']:5.1f}%  ({int(row['n_videos'])} videos)")

        if shorts_rate is not None and overall_non_shorts is not None:
            diff = abs(shorts_rate - overall_non_shorts)
            if diff > 10:
                print(f"\n  [CANH BAO] Shorts viral_rate ({shorts_rate:.1f}%) lech {diff:.1f}% so voi trung binh cac loai khac.")
    else:
        print("  [!] Khong co cot video_length_category — bo qua kiem tra.")

    # ── KIỂM TRA 4 — Data leakage ─────────────────────────────────────────────
    print("\n[KIỂM TRA 4] Kiểm tra data leakage tiềm ẩn")
    print("-" * 50)

    leakage_ok = True
    df_leak = df_labeled.merge(
        df_engagement[["video_id", "is_potentially_viral", "engagement_level"]],
        on="video_id",
        how="left",
    )

    correlation_targets = {}
    if "is_potentially_viral" in df_leak.columns:
        correlation_targets["is_potentially_viral"] = df_leak["is_potentially_viral"].astype(int)
    if "engagement_level" in df_leak.columns:
        level_map = {"high": 2, "medium": 1, "low": 0}
        correlation_targets["engagement_level"] = df_leak["engagement_level"].map(level_map).fillna(0)

    for col_name, col_series in correlation_targets.items():
        corr = df_leak["is_viral"].corr(col_series)
        if abs(corr) > 0.7:
            tag = "[CANH BAO]"
            leakage_ok = False
        else:
            tag = "[OK]     "
        print(f"  {tag}  is_viral vs {col_name:<30} : corr = {corr:.2f}")

    checks["Data leakage check"] = (
        "PASS  → Khong phat hien" if leakage_ok else "CANH BAO  → Correlation > 0.7 phat hien"
    )

    # ── KIỂM TRA 5 — Tổng kết ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(" KET QUA KIEM TRA CHAT LUONG NHAN")
    print("=" * 60)
    for check_name, result in checks.items():
        tag = "[PASS]    " if result.startswith("PASS") else "[CANH BAO]"
        # Remove the embedded tag from result for cleaner display
        clean_result = result.replace("PASS  → ", "").replace("CANH BAO  → ", "")
        print(f"  {tag} {check_name:<30} : {clean_result}")
    print("=" * 60)
    print()


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 4 — HÀM TIỆN ÍCH
# ═════════════════════════════════════════════════════════════════════════════

def get_label_stats(df_labeled: pd.DataFrame) -> dict:
    """
    Return a compact summary dict for use by downstream modules.

    Parameters
    ----------
    df_labeled : pd.DataFrame
        Output of define_viral_label().

    Returns
    -------
    dict
        Keys: total_videos, viral_count, viral_rate, strategy,
              threshold, min_days.
    """
    if df_labeled.empty or "is_viral" not in df_labeled.columns:
        return {
            "total_videos": 0,
            "viral_count":  0,
            "viral_rate":   0.0,
            "strategy":     "unknown",
            "threshold":    0.0,
            "min_days":     0,
        }

    total   = len(df_labeled)
    viral   = int(df_labeled["is_viral"].sum())
    rate    = viral / total * 100 if total > 0 else 0.0
    strat   = df_labeled["label_strategy"].iloc[0] if "label_strategy" in df_labeled.columns else "unknown"

    # Best-effort threshold from view_ratio of viral videos
    if "view_ratio" in df_labeled.columns and viral > 0:
        threshold = float(df_labeled.loc[df_labeled["is_viral"] == 1, "view_ratio"].min())
    else:
        threshold = 0.0

    min_days = int(df_labeled["days_since_published"].min()) if "days_since_published" in df_labeled.columns else 0

    return {
        "total_videos": total,
        "viral_count":  viral,
        "viral_rate":   round(rate, 2),
        "strategy":     strat,
        "threshold":    round(threshold, 2),
        "min_days":     min_days,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 5 — CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from ml.data_loader import load_all_intermediate_data

    # 1. Load dữ liệu
    print("\n>>> BƯỚC 1: Load dữ liệu từ BigQuery …")
    data = load_all_intermediate_data()
    df_engagement = data["engagement"]
    df_channels   = data["channels"]

    # 2. EDA trước khi định nghĩa nhãn
    print("\n>>> BƯỚC 2: Phân tích ứng viên nhãn …")
    analyze_label_candidates(df_engagement, df_channels)

    # 3. Định nghĩa nhãn viral
    print("\n>>> BƯỚC 3: Định nghĩa nhãn viral …")
    df_labeled = define_viral_label(
        df_engagement=df_engagement,
        df_channels=df_channels,
        strategy="auto",
    )

    # 4. Kiểm tra chất lượng nhãn
    print("\n>>> BƯỚC 4: Kiểm tra chất lượng nhãn …")
    validate_label_quality(df_labeled, df_engagement)

    # 5. Stats tóm tắt
    stats = get_label_stats(df_labeled)
    print("\n>>> BƯỚC 5: Label stats:")
    for k, v in stats.items():
        print(f"    {k:<18} : {v}")

    # 6. Xem thử kết quả
    print("\n>>> Mẫu 5 video VIRAL (is_viral = 1):")
    viral_sample = df_labeled[df_labeled["is_viral"] == 1].head(5)
    print(viral_sample[["video_id", "channel_id", "view_count", "view_ratio",
                         "velocity_score", "is_viral", "label_strategy"]].to_string(index=False))

    print("\n>>> Mẫu 5 video KHÔNG VIRAL (is_viral = 0):")
    non_viral_sample = df_labeled[df_labeled["is_viral"] == 0].head(5)
    print(non_viral_sample[["video_id", "channel_id", "view_count", "view_ratio",
                             "velocity_score", "is_viral", "label_strategy"]].to_string(index=False))

    print("\n>>> df_labeled shape:", df_labeled.shape)
    print(">>> Dtypes:\n", df_labeled.dtypes.to_string())
