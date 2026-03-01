"""
Data loader for the YouTube viral video prediction ML pipeline.

Fetches data from the three Intermediate-layer tables in BigQuery:
  - int_videos__enhanced      : per-video enriched metadata
  - int_engagement_metrics    : per-video computed engagement KPIs
  - int_channel_summary       : aggregated channel-level statistics

Usage (standalone):
    python -m ml.data_loader1
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

from ml.config import (
    BQ_INTERMEDIATE_DATASET,
    BQ_LOCATION,
    CREDENTIALS_PATH,
    GCP_PROJECT_ID,
    INT_CHANNEL_SUMMARY_TABLE,
    INT_ENGAGEMENT_METRICS_TABLE,
    INT_VIDEOS_ENHANCED_TABLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── BigQuery client ──────────────────────────────────────────────────────────

def _build_client() -> bigquery.Client:
    """Create an authenticated BigQuery client from a service-account key file."""
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )
    return bigquery.Client(
        project=GCP_PROJECT_ID,
        credentials=credentials,
        location=BQ_LOCATION,
    )


# ── Generic helpers ──────────────────────────────────────────────────────────

def _full_table(table: str) -> str:
    return f"`{GCP_PROJECT_ID}.{BQ_INTERMEDIATE_DATASET}.{table}`"


def _run_query(client: bigquery.Client, sql: str) -> pd.DataFrame:
    logger.debug("Running query:\n%s", sql)
    return client.query(sql).result().to_dataframe()


# ── Individual table loaders ─────────────────────────────────────────────────

def load_videos_enhanced(
    client: bigquery.Client,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load int_videos__enhanced — enriched video metadata joined with channel info.

    Returns one row per video with columns including:
        video_id, channel_id, title, published_at, view_count, like_count,
        comment_count, duration_seconds, video_length_category,
        channel_name, channel_subscribers, days_since_published, …
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    sql = f"""
        SELECT *
        FROM {_full_table(INT_VIDEOS_ENHANCED_TABLE)}
        {limit_clause}
    """
    df = _run_query(client, sql)
    logger.info("int_videos__enhanced  → %d rows, %d columns", len(df), df.shape[1])
    return df


def load_engagement_metrics(
    client: bigquery.Client,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load int_engagement_metrics — per-video engagement KPIs.

    Returns one row per video with columns including:
        video_id, channel_id, engagement_score, like_rate_pct,
        comment_rate_pct, avg_views_per_day, engagement_level,
        is_potentially_viral, …
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    sql = f"""
        SELECT *
        FROM {_full_table(INT_ENGAGEMENT_METRICS_TABLE)}
        {limit_clause}
    """
    df = _run_query(client, sql)
    logger.info("int_engagement_metrics → %d rows, %d columns", len(df), df.shape[1])
    return df


def load_channel_summary(
    client: bigquery.Client,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load int_channel_summary — aggregated channel-level statistics.

    Returns one row per channel with columns including:
        channel_id, channel_name, subscriber_count, total_videos_crawled,
        avg_views_per_video, avg_like_rate_pct, avg_days_between_uploads, …
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    sql = f"""
        SELECT *
        FROM {_full_table(INT_CHANNEL_SUMMARY_TABLE)}
        {limit_clause}
    """
    df = _run_query(client, sql)
    logger.info("int_channel_summary    → %d rows, %d columns", len(df), df.shape[1])
    return df


# ── Main loader: fetch all three tables ─────────────────────────────────────

def load_all_intermediate_data(
    videos_limit: Optional[int] = None,
    engagement_limit: Optional[int] = None,
    channel_limit: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """
    Connect to BigQuery and pull all three Intermediate-layer tables.

    Parameters
    ----------
    videos_limit : int, optional
        Max rows to fetch from int_videos__enhanced (default: all rows).
    engagement_limit : int, optional
        Max rows to fetch from int_engagement_metrics (default: all rows).
    channel_limit : int, optional
        Max rows to fetch from int_channel_summary (default: all rows).

    Returns
    -------
    dict with keys:
        "videos"     → DataFrame from int_videos__enhanced
        "engagement" → DataFrame from int_engagement_metrics
        "channels"   → DataFrame from int_channel_summary
    """
    logger.info("=" * 60)
    logger.info("Connecting to BigQuery  [project: %s]", GCP_PROJECT_ID)
    logger.info("Dataset : %s", BQ_INTERMEDIATE_DATASET)
    logger.info("=" * 60)

    client = _build_client()

    df_videos = load_videos_enhanced(client, limit=videos_limit)
    df_engagement = load_engagement_metrics(client, limit=engagement_limit)
    df_channels = load_channel_summary(client, limit=channel_limit)

    _print_summary(df_videos, df_engagement, df_channels)

    return {
        "videos": df_videos,
        "engagement": df_engagement,
        "channels": df_channels,
    }


# ── Summary report ───────────────────────────────────────────────────────────

def _print_summary(
    df_videos: pd.DataFrame,
    df_engagement: pd.DataFrame,
    df_channels: pd.DataFrame,
) -> None:
    """Print a concise data-quality summary to the console."""

    n_videos_enhanced = len(df_videos)
    n_videos_engagement = len(df_engagement)
    n_channels_from_videos = df_videos["channel_id"].nunique() if "channel_id" in df_videos.columns else "N/A"
    n_channels_summary = len(df_channels)

    # Viral label distribution (if available)
    viral_info = ""
    if "is_potentially_viral" in df_engagement.columns and n_videos_engagement > 0:
        viral_count = df_engagement["is_potentially_viral"].sum()
        viral_pct = viral_count / n_videos_engagement * 100
        viral_info = f"  Viral label (is_potentially_viral = True) : {int(viral_count):,} / {n_videos_engagement:,}  ({viral_pct:.1f}%)"

    print()
    print("=" * 60)
    print(" DATA LOAD SUMMARY  —  Intermediate Layer")
    print("=" * 60)
    print(f"  int_videos__enhanced")
    print(f"    Videos   : {n_videos_enhanced:,}")
    print(f"    Channels : {n_channels_from_videos:,}  (distinct channel_id)")
    print()
    print(f"  int_engagement_metrics")
    print(f"    Videos   : {n_videos_engagement:,}")
    if viral_info:
        print(viral_info)
    print()
    print(f"  int_channel_summary")
    print(f"    Channels : {n_channels_summary:,}")
    print("=" * 60)
    print()


# ── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_all_intermediate_data()

    # Quick sanity-check: print first 3 rows of each DataFrame
    for name, df in data.items():
        print(f"\n--- {name} (first 3 rows) ---")
        print(df.head(3).to_string(index=False))
