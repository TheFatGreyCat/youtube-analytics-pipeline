"""
Configuration for the ML module.1
Reads environment variables from .env file at the project root.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Project root & .env ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Google Cloud / BigQuery ──────────────────────────────────────────────────
GCP_PROJECT_ID: str | None = os.getenv("GCP_PROJECT_ID")
CREDENTIALS_PATH: str | None = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Resolve relative paths to absolute
if CREDENTIALS_PATH and not os.path.isabs(CREDENTIALS_PATH):
    CREDENTIALS_PATH = str(PROJECT_ROOT / CREDENTIALS_PATH)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH


def validate_ml_config() -> None:
    """Raise if required environment variables are missing."""
    missing = [k for k, v in {"GCP_PROJECT_ID": GCP_PROJECT_ID, "GOOGLE_APPLICATION_CREDENTIALS": CREDENTIALS_PATH}.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your values."
        )

# ── BigQuery dataset names ───────────────────────────────────────────────────
BQ_INTERMEDIATE_DATASET: str = "intermediate"

# Intermediate tables used for ML feature engineering
INT_VIDEOS_ENHANCED_TABLE: str = "int_videos__enhanced"
INT_ENGAGEMENT_METRICS_TABLE: str = "int_engagement_metrics"
INT_CHANNEL_SUMMARY_TABLE: str = "int_channel_summary"

# ── BigQuery location ────────────────────────────────────────────────────────
BQ_LOCATION: str = "asia-southeast1"
