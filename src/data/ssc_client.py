"""NASA SSC (Satellite Situation Center) API client.

Fetches spacecraft position data via the sscws Python library and caches
results as Parquet files for efficient reuse.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sscws.sscws import SscWs
from sscws.coordinates import CoordinateSystem
from tqdm import tqdm

# Map config strings to sscws enums
COORD_MAP = {
    "Gse": CoordinateSystem.GSE,
    "Geo": CoordinateSystem.GEO,
    "Gsm": CoordinateSystem.GSM,
    "Gm": CoordinateSystem.GM,
    "Sm": CoordinateSystem.SM,
    "GeiTod": CoordinateSystem.GEI_TOD,
    "GeiJ2000": CoordinateSystem.GEI_J_2000,
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class SSCClient:
    """Client for fetching and caching spacecraft position data."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.ssc = SscWs()
        self.raw_dir = Path(self.config["data"]["raw_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def list_observatories(self) -> pd.DataFrame:
        """Get all available spacecraft from SSC."""
        result = self.ssc.get_observatories()
        obs = result["Observatory"]
        rows = []
        for o in obs:
            rows.append({
                "id": o["Id"],
                "name": o["Name"],
                "start_time": o["StartTime"][0] if isinstance(o["StartTime"], list) else o["StartTime"],
                "end_time": o["EndTime"][0] if isinstance(o["EndTime"], list) else o["EndTime"],
            })
        return pd.DataFrame(rows)

    def fetch_positions(
        self,
        spacecraft_id: str,
        start_date: str,
        end_date: str,
        coord_systems: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch spacecraft positions, chunking by month to avoid timeouts.

        Args:
            spacecraft_id: SSC observatory ID (e.g. 'iss', 'dscovr', 'mms1')
            start_date: Start date as 'YYYY-MM-DD'
            end_date: End date as 'YYYY-MM-DD'
            coord_systems: List of coordinate system names (default from config)

        Returns:
            DataFrame with columns: time, x_gse, y_gse, z_gse, x_geo, y_geo, z_geo, lat_*, lon_*
        """
        if coord_systems is None:
            coord_systems = self.config["coordinate_systems"]

        # Convert string coord names to enum values
        coord_enums = [COORD_MAP[cs] for cs in coord_systems]

        cache_file = self.raw_dir / f"{spacecraft_id}_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            print(f"Loading cached data: {cache_file}")
            return pd.read_parquet(cache_file)

        chunk_days = self.config["data"]["chunk_days"]
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_frames = []
        current = start

        pbar = tqdm(
            total=(end - start).days,
            desc=f"Fetching {spacecraft_id}",
            unit="days",
        )

        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            time_range = [
                current.strftime("%Y-%m-%dT00:00:00Z"),
                chunk_end.strftime("%Y-%m-%dT00:00:00Z"),
            ]

            try:
                result = self.ssc.get_locations(
                    [spacecraft_id],
                    time_range,
                    coord_enums,
                )
                df = self._parse_location_result(result)
                if df is not None and len(df) > 0:
                    all_frames.append(df)
            except Exception as e:
                print(f"Warning: Failed chunk {current} - {chunk_end}: {e}")

            pbar.update((chunk_end - current).days)
            current = chunk_end

        pbar.close()

        if not all_frames:
            print(f"No data retrieved for {spacecraft_id}")
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

        combined.to_parquet(cache_file, index=False)
        print(f"Cached {len(combined)} positions to {cache_file}")

        return combined

    def _parse_location_result(self, result: dict) -> pd.DataFrame | None:
        """Parse sscws get_locations response into a DataFrame.

        The sscws library returns:
          result['Data'] -> numpy array of dicts, one per spacecraft
          Each dict has:
            'Time': array of datetime objects
            'Coordinates': array of dicts, one per coordinate system
              Each coordinate dict has:
                'CoordinateSystem': CoordinateSystem enum
                'X', 'Y', 'Z': arrays of floats (km)
                'Latitude', 'Longitude': arrays of floats (degrees)
        """
        if result is None:
            return None

        try:
            status = result.get("HttpStatus", 200)
            if status != 200:
                return None

            data_array = result.get("Data", None)
            if data_array is None or len(data_array) == 0:
                return None

            # First spacecraft in the response
            spacecraft_data = data_array[0]

            time_data = spacecraft_data.get("Time", [])
            if len(time_data) == 0:
                return None

            rows = {"time": pd.to_datetime(time_data, utc=True)}

            coords_array = spacecraft_data.get("Coordinates", [])
            for coord in coords_array:
                cs = coord.get("CoordinateSystem")
                # Get the lowercase name (e.g. 'gse', 'geo')
                if hasattr(cs, 'name'):
                    sys_name = cs.name.lower()
                elif hasattr(cs, 'value'):
                    sys_name = cs.value.lower()
                else:
                    sys_name = str(cs).lower()

                for field in ["X", "Y", "Z", "Latitude", "Longitude"]:
                    values = coord.get(field, np.array([]))
                    if len(values) > 0:
                        col_name = f"{field.lower()}_{sys_name}"
                        # Shorten field names
                        col_name = col_name.replace("latitude", "lat").replace("longitude", "lon")
                        rows[col_name] = values

            return pd.DataFrame(rows)

        except Exception as e:
            print(f"Warning: Could not parse result: {e}")
            return None

    def fetch_all_spacecraft(self) -> dict[str, pd.DataFrame]:
        """Fetch data for all spacecraft defined in config."""
        results = {}
        for name, sc_config in self.config["spacecraft"].items():
            print(f"\n{'='*60}")
            print(f"Fetching {name} ({sc_config['orbit_type']})")
            print(f"{'='*60}")
            df = self.fetch_positions(
                sc_config["id"],
                sc_config["start_date"],
                sc_config["end_date"],
            )
            results[name] = df
            print(f"  -> {len(df)} data points")
        return results


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent.parent)
    client = SSCClient()

    # List available observatories
    print("Available observatories:")
    obs = client.list_observatories()
    print(f"  Total: {len(obs)}")
    print(obs.head(10))

    # Fetch all configured spacecraft
    data = client.fetch_all_spacecraft()
    for name, df in data.items():
        print(f"\n{name}: {df.shape}")
        if len(df) > 0:
            print(df.head())
