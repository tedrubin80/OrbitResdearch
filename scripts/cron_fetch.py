#!/usr/bin/env python3
"""Cron-ready data fetcher with logging, S3 backup, and incremental updates.

Designed to run unattended via crontab. Fetches spacecraft position data
and solar wind data, caches locally as Parquet, and backs up to S3.

Usage:
    python scripts/cron_fetch.py                    # Full fetch + S3 backup
    python scripts/cron_fetch.py --no-s3            # Skip S3 backup
    python scripts/cron_fetch.py --spacecraft iss   # Single spacecraft
    python scripts/cron_fetch.py --extend           # Extend data to today
"""

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

import yaml

# Logging setup
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"fetch_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("orbit-fetch")


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def fetch_spacecraft_data(config: dict, spacecraft_filter: str = None):
    """Fetch spacecraft position data from NASA SSC."""
    from src.data.ssc_client import SSCClient

    ssc = SSCClient()

    spacecraft = config["spacecraft"]
    if spacecraft_filter:
        spacecraft = {k: v for k, v in spacecraft.items() if k == spacecraft_filter}

    results = {}
    for name, sc_config in spacecraft.items():
        log.info(f"Fetching {name} ({sc_config['orbit_type']}) "
                 f"{sc_config['start_date']} to {sc_config['end_date']}")
        try:
            df = ssc.fetch_positions(
                sc_config["id"],
                sc_config["start_date"],
                sc_config["end_date"],
            )
            results[name] = df
            log.info(f"  {name}: {len(df)} data points, "
                     f"columns: {list(df.columns)}")
            if len(df) > 0:
                log.info(f"  Time range: {df['time'].min()} to {df['time'].max()}")
        except Exception as e:
            log.error(f"  FAILED {name}: {e}")
            log.error(traceback.format_exc())

    return results


def fetch_solar_wind_data(config: dict):
    """Fetch OMNI solar wind data from CDAWeb."""
    from src.data.solar_wind import SolarWindClient

    solar = SolarWindClient()

    # Get date range covering all spacecraft
    starts = [sc["start_date"] for sc in config["spacecraft"].values()]
    ends = [sc["end_date"] for sc in config["spacecraft"].values()]
    start = min(starts)
    end = max(ends)

    log.info(f"Fetching solar wind data {start} to {end}")
    try:
        df = solar.fetch_solar_wind(start, end)
        log.info(f"  Solar wind: {len(df)} records")
        if len(df) > 0:
            log.info(f"  Time range: {df['time'].min()} to {df['time'].max()}")
            log.info(f"  Columns: {list(df.columns)}")
            # Log basic stats for key parameters
            for col in ["flow_speed", "bz_gse", "kp"]:
                if col in df.columns:
                    log.info(f"  {col}: mean={df[col].mean():.2f}, "
                             f"std={df[col].std():.2f}, "
                             f"missing={df[col].isna().sum()}")
        return df
    except Exception as e:
        log.error(f"  FAILED solar wind: {e}")
        log.error(traceback.format_exc())
        return None


def extend_to_today(config: dict) -> dict:
    """Update config end dates to today for continuous data collection."""
    today = datetime.now().strftime("%Y-%m-%d")
    for name in config["spacecraft"]:
        config["spacecraft"][name]["end_date"] = today
    log.info(f"Extended all end dates to {today}")
    return config


def backup_to_s3():
    """Upload raw data files to S3-compatible object storage."""
    import boto3
    from botocore.config import Config as BotoConfig

    access_key = os.getenv("S3_ACCESS_KEY")
    secret_key = os.getenv("S3_SECRET_KEY")
    endpoint = os.getenv("S3_ENDPOINT")
    region = os.getenv("S3_REGION", "eu-central")
    bucket = os.getenv("S3_BUCKET", "orbit-research")

    if not all([access_key, secret_key, endpoint]):
        log.warning("S3 credentials not configured, skipping backup")
        return

    log.info(f"Backing up to S3: {endpoint}/{bucket}")

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=BotoConfig(signature_version="s3v4"),
        )

        # Create bucket if needed
        try:
            s3.head_bucket(Bucket=bucket)
        except Exception:
            log.info(f"Creating bucket: {bucket}")
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )

        # Upload all parquet files from data/raw/
        raw_dir = PROJECT_ROOT / "data" / "raw"
        if not raw_dir.exists():
            log.warning("No data/raw/ directory found")
            return

        uploaded = 0
        for parquet_file in raw_dir.glob("*.parquet"):
            s3_key = f"data/raw/{parquet_file.name}"
            log.info(f"  Uploading {parquet_file.name} -> s3://{bucket}/{s3_key}")
            s3.upload_file(str(parquet_file), bucket, s3_key)
            uploaded += 1

        # Also upload logs
        for log_file in LOG_DIR.glob("*.log"):
            s3_key = f"logs/{log_file.name}"
            s3.upload_file(str(log_file), bucket, s3_key)

        log.info(f"  Backed up {uploaded} data files to S3")

    except Exception as e:
        log.error(f"  S3 backup failed: {e}")
        log.error(traceback.format_exc())


def generate_data_report(orbit_results: dict, solar_df) -> str:
    """Generate a summary report of fetched data."""
    lines = [
        "=" * 60,
        f"DATA FETCH REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]

    total_points = 0
    for name, df in orbit_results.items():
        n = len(df)
        total_points += n
        if n > 0:
            duration = (df["time"].max() - df["time"].min()).days
            lines.append(f"{name:>10}: {n:>10,} points | {duration} days")
        else:
            lines.append(f"{name:>10}: NO DATA")

    if solar_df is not None and len(solar_df) > 0:
        total_points += len(solar_df)
        lines.append(f"{'solar':>10}: {len(solar_df):>10,} points")

    lines.extend([
        "",
        f"Total data points: {total_points:,}",
        f"Raw data size: {sum(f.stat().st_size for f in (PROJECT_ROOT / 'data' / 'raw').glob('*.parquet')) / 1e6:.1f} MB"
        if (PROJECT_ROOT / "data" / "raw").exists() else "Raw data dir not found",
        "",
    ])

    report = "\n".join(lines)
    log.info(report)

    # Save report
    report_file = PROJECT_ROOT / "data" / "fetch_report.txt"
    with open(report_file, "w") as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Cron-ready orbit data fetcher")
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 backup")
    parser.add_argument("--spacecraft", type=str, default=None,
                        help="Fetch specific spacecraft only")
    parser.add_argument("--solar-only", action="store_true",
                        help="Fetch only solar wind data")
    parser.add_argument("--orbit-only", action="store_true",
                        help="Fetch only orbit data")
    parser.add_argument("--extend", action="store_true",
                        help="Extend end date to today")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("ORBIT DATA FETCH STARTING")
    log.info("=" * 60)

    config = load_config()

    if args.extend:
        config = extend_to_today(config)

    orbit_results = {}
    solar_df = None

    # Fetch orbit data
    if not args.solar_only:
        orbit_results = fetch_spacecraft_data(config, args.spacecraft)

    # Fetch solar wind data
    if not args.orbit_only:
        solar_df = fetch_solar_wind_data(config)

    # Generate report
    generate_data_report(orbit_results, solar_df)

    # S3 backup
    if not args.no_s3:
        backup_to_s3()

    log.info("FETCH COMPLETE")


if __name__ == "__main__":
    main()
