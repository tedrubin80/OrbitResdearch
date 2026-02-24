#!/usr/bin/env python3
"""Bulk data download script for orbit and solar wind data.

Usage:
    python scripts/fetch_data.py                     # Fetch all configured data
    python scripts/fetch_data.py --spacecraft iss     # Fetch specific spacecraft
    python scripts/fetch_data.py --solar-only         # Fetch only solar wind data
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.data.ssc_client import SSCClient
from src.data.solar_wind import SolarWindClient


def main():
    parser = argparse.ArgumentParser(description="Fetch orbit and solar wind data")
    parser.add_argument("--spacecraft", type=str, default=None,
                        help="Specific spacecraft ID to fetch (default: all)")
    parser.add_argument("--solar-only", action="store_true",
                        help="Fetch only solar wind data")
    parser.add_argument("--orbit-only", action="store_true",
                        help="Fetch only orbit data")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    if not args.solar_only:
        print("=" * 60)
        print("FETCHING SPACECRAFT POSITION DATA")
        print("=" * 60)
        ssc = SSCClient(args.config)

        if args.spacecraft:
            config = ssc.config["spacecraft"].get(args.spacecraft)
            if config is None:
                print(f"Unknown spacecraft: {args.spacecraft}")
                print(f"Available: {list(ssc.config['spacecraft'].keys())}")
                sys.exit(1)
            df = ssc.fetch_positions(config["id"], config["start_date"], config["end_date"])
            print(f"\n{args.spacecraft}: {df.shape}")
        else:
            data = ssc.fetch_all_spacecraft()
            for name, df in data.items():
                print(f"  {name}: {df.shape}")

    if not args.orbit_only:
        print("\n" + "=" * 60)
        print("FETCHING SOLAR WIND DATA")
        print("=" * 60)
        solar = SolarWindClient(args.config)
        df = solar.fetch_for_date_range()
        print(f"\nSolar wind: {df.shape}")
        if len(df) > 0:
            print(f"  Time range: {df['time'].min()} to {df['time'].max()}")
            print(f"  Columns: {list(df.columns)}")

    print("\nDone! Data cached in data/raw/")


if __name__ == "__main__":
    main()
