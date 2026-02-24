"""Solar wind data fetcher using NASA CDAWeb (OMNI database).

Fetches solar wind parameters (speed, density, IMF) at 1-min resolution
and geomagnetic indices (Kp, Dst, AE) at hourly resolution, then merges them.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from cdasws import CdasWs
from tqdm import tqdm


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class SolarWindClient:
    """Client for fetching and caching OMNI solar wind data."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.cdas = CdasWs()
        self.raw_dir = Path(self.config["data"]["raw_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch_solar_wind(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch OMNI solar wind data (1-min plasma + hourly indices), merged.

        Args:
            start_date: Start date as 'YYYY-MM-DD'
            end_date: End date as 'YYYY-MM-DD'

        Returns:
            DataFrame with columns: time, bx_gse, by_gse, bz_gse,
            flow_speed, proton_density, kp, dst, ae
        """
        cache_file = self.raw_dir / f"solar_wind_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            print(f"Loading cached solar wind data: {cache_file}")
            return pd.read_parquet(cache_file)

        sw_config = self.config["solar_wind"]

        # Fetch 1-minute plasma/IMF data
        print("Fetching 1-minute OMNI plasma/IMF data...")
        plasma_df = self._fetch_dataset(
            sw_config["dataset"],
            sw_config["variables"],
            start_date,
            end_date,
        )

        # Fetch hourly geomagnetic indices
        print("Fetching hourly geomagnetic indices...")
        indices_df = self._fetch_dataset(
            sw_config["indices_dataset"],
            sw_config["indices_variables"],
            start_date,
            end_date,
        )

        # Merge: forward-fill hourly indices into 1-min data
        if plasma_df is not None and len(plasma_df) > 0:
            combined = plasma_df
            if indices_df is not None and len(indices_df) > 0:
                # Rename index columns for clarity
                rename_map = {"KP1800": "kp", "DST1800": "dst", "AE1800": "ae"}
                indices_df = indices_df.rename(columns={
                    k: v for k, v in rename_map.items() if k in indices_df.columns
                })
                # Keep only time + renamed columns
                keep_cols = ["time"] + [v for v in rename_map.values() if v in indices_df.columns]
                indices_df = indices_df[[c for c in keep_cols if c in indices_df.columns]]
                combined = pd.merge_asof(
                    combined.sort_values("time"),
                    indices_df.sort_values("time"),
                    on="time",
                    direction="backward",
                )
        elif indices_df is not None and len(indices_df) > 0:
            combined = indices_df
        else:
            print("No solar wind data retrieved")
            return pd.DataFrame()

        # Standardize column names
        rename_map = {
            "BX_GSE": "bx_gse", "BY_GSE": "by_gse", "BZ_GSE": "bz_gse",
        }
        combined = combined.rename(columns={
            k: v for k, v in rename_map.items() if k in combined.columns
        })

        # Drop extra columns from CDAWeb (Epoch, duplicate index columns)
        keep_cols = ["time", "bx_gse", "by_gse", "bz_gse",
                     "flow_speed", "proton_density", "kp", "dst", "ae"]
        combined = combined[[c for c in keep_cols if c in combined.columns]]

        # Clean fill values
        combined = self._clean_fill_values(combined)

        combined.to_parquet(cache_file, index=False)
        print(f"Cached {len(combined)} solar wind records to {cache_file}")

        return combined

    def _fetch_dataset(
        self,
        dataset: str,
        variables: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        """Fetch a single CDAWeb dataset, chunking by month."""
        chunk_days = self.config["data"]["chunk_days"]
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_frames = []
        current = start

        pbar = tqdm(
            total=(end - start).days,
            desc=f"  {dataset}",
            unit="days",
        )

        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)

            try:
                status, data = self.cdas.get_data(
                    dataset,
                    variables,
                    current,
                    chunk_end,
                )

                if data is not None:
                    df = self._parse_cdas_result(data)
                    if df is not None and len(df) > 0:
                        all_frames.append(df)
            except Exception as e:
                print(f"\n  Warning: {dataset} chunk {current.date()}-{chunk_end.date()}: {e}")

            pbar.update((chunk_end - current).days)
            current = chunk_end

        pbar.close()

        if not all_frames:
            return None

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        return combined

    def _parse_cdas_result(self, data) -> pd.DataFrame | None:
        """Parse CDAWeb xarray response into a DataFrame."""
        try:
            if hasattr(data, "to_dataframe"):
                df = data.to_dataframe().reset_index()
                # Find and rename the time/epoch column
                for col in df.columns:
                    if "epoch" in col.lower() or col.lower() == "time":
                        df = df.rename(columns={col: "time"})
                        break
                # Convert time column
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], utc=True)
                return df
            return None
        except Exception as e:
            print(f"  Warning: parse error: {e}")
            return None

    def _clean_fill_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace OMNI fill values with NaN."""
        fill_thresholds = {
            "flow_speed": 99999.0,
            "proton_density": 999.0,
            "bx_gse": 9999.0,
            "by_gse": 9999.0,
            "bz_gse": 9999.0,
            "kp": 90.0,
            "dst": 99999.0,
            "ae": 9999.0,
        }

        for col, threshold in fill_thresholds.items():
            if col in df.columns:
                df.loc[df[col].abs() >= threshold, col] = np.nan

        return df

    def fetch_for_date_range(self) -> pd.DataFrame:
        """Fetch solar wind data covering all configured spacecraft date ranges."""
        all_starts = []
        all_ends = []
        for sc_config in self.config["spacecraft"].values():
            all_starts.append(sc_config["start_date"])
            all_ends.append(sc_config["end_date"])

        start = min(all_starts)
        end = max(all_ends)
        return self.fetch_solar_wind(start, end)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent.parent)
    client = SolarWindClient()
    df = client.fetch_for_date_range()
    print(f"\nSolar wind data shape: {df.shape}")
    if len(df) > 0:
        print(df.head())
        print(f"\nColumn stats:\n{df.describe()}")
