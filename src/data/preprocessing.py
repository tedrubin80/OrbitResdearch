"""Preprocessing pipeline for orbit and solar wind data.

Handles normalization, velocity derivation, sliding window creation,
temporal train/val/test splits, and multi-modal alignment.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class OrbitPreprocessor:
    """Preprocesses spacecraft position data for ML training."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.processed_dir = Path(self.config["data"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {}  # Per-spacecraft normalization stats

    def preprocess(self, df: pd.DataFrame, spacecraft_id: str) -> pd.DataFrame:
        """Full preprocessing: derive velocity, normalize, handle gaps.

        Args:
            df: Raw position data with time, x_gse, y_gse, z_gse columns
            spacecraft_id: ID for caching normalization stats

        Returns:
            Preprocessed DataFrame with position and velocity columns
        """
        df = df.copy()
        df = df.sort_values("time").reset_index(drop=True)

        # Ensure numeric columns
        pos_cols = [c for c in df.columns if c.startswith(("x_", "y_", "z_"))]
        for col in pos_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Derive velocity from finite differences (km/s)
        dt = df["time"].diff().dt.total_seconds()
        for axis in ["x_gse", "y_gse", "z_gse"]:
            if axis in df.columns:
                vel_col = axis.replace("x_", "vx_").replace("y_", "vy_").replace("z_", "vz_")
                df[vel_col] = df[axis].diff() / dt

        # Drop first row (no velocity) and any NaN rows
        df = df.iloc[1:].dropna(subset=[c for c in df.columns if c != "time"])
        df = df.reset_index(drop=True)

        # Remove large gaps (> 10 minutes between points indicates missing data)
        time_diff = df["time"].diff().dt.total_seconds()
        gap_mask = time_diff > 600  # 10 min threshold
        df["segment_id"] = gap_mask.cumsum()

        # Compute and store normalization statistics
        feature_cols = self._get_feature_cols(df)
        self.stats[spacecraft_id] = {
            "mean": df[feature_cols].mean().to_dict(),
            "std": df[feature_cols].std().to_dict(),
        }

        # Normalize features (zero mean, unit variance)
        for col in feature_cols:
            mean = self.stats[spacecraft_id]["mean"][col]
            std = self.stats[spacecraft_id]["std"][col]
            if std > 0:
                df[f"{col}_norm"] = (df[col] - mean) / std
            else:
                df[f"{col}_norm"] = 0.0

        return df

    def create_sliding_windows(
        self,
        df: pd.DataFrame,
        input_hours: int = None,
        horizon_hours: int = 6,
        stride_hours: int = 1,
        subsample: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sliding windows for sequence-to-sequence training.

        Args:
            df: Preprocessed DataFrame
            input_hours: Hours of input context (default from config)
            horizon_hours: Hours to predict ahead
            stride_hours: Stride between windows
            subsample: Take every Nth point within windows (e.g. 10 = 10-min res from 1-min data)

        Returns:
            Tuple of (inputs, targets, timestamps):
              inputs: (N, input_steps, features)
              targets: (N, horizon_steps, 3)  # x, y, z positions
              timestamps: (N,) start times for each window
        """
        if input_hours is None:
            input_hours = self.config["model"]["input_hours"]

        time_res = self.config["model"]["time_resolution_minutes"]
        input_steps = (input_hours * 60) // time_res
        horizon_steps = (horizon_hours * 60) // time_res
        stride_steps = (stride_hours * 60) // time_res

        feature_cols = [c for c in df.columns if c.endswith("_norm")]
        target_cols = ["x_gse_norm", "y_gse_norm", "z_gse_norm"]

        # Filter to only available target columns
        target_cols = [c for c in target_cols if c in df.columns]
        feature_cols = [c for c in feature_cols if c in df.columns]

        if not feature_cols or not target_cols:
            raise ValueError(f"Missing required columns. Available: {list(df.columns)}")

        inputs_list = []
        targets_list = []
        times_list = []

        # Process each continuous segment separately
        for _, segment in df.groupby("segment_id"):
            if len(segment) < input_steps + horizon_steps:
                continue

            features = segment[feature_cols].values
            targets = segment[target_cols].values
            timestamps = segment["time"].values

            for i in range(0, len(segment) - input_steps - horizon_steps, stride_steps):
                inp = features[i : i + input_steps]
                tgt = targets[i + input_steps : i + input_steps + horizon_steps]
                if subsample > 1:
                    inp = inp[::subsample]
                    tgt = tgt[::subsample]
                inputs_list.append(inp)
                targets_list.append(tgt)
                times_list.append(timestamps[i])

        if not inputs_list:
            raise ValueError("No valid windows created. Check data continuity and window size.")

        return (
            np.array(inputs_list, dtype=np.float32),
            np.array(targets_list, dtype=np.float32),
            np.array(times_list),
        )

    def temporal_split(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Split data chronologically into train/val/test.

        Returns:
            Dict with 'train', 'val', 'test' keys, each containing (inputs, targets)
        """
        n = len(inputs)
        train_end = int(n * self.config["training"]["train_split"])
        val_end = train_end + int(n * self.config["training"]["val_split"])

        return {
            "train": (inputs[:train_end], targets[:train_end]),
            "val": (inputs[train_end:val_end], targets[train_end:val_end]),
            "test": (inputs[val_end:], targets[val_end:]),
        }

    def denormalize(self, predictions: np.ndarray, spacecraft_id: str) -> np.ndarray:
        """Convert normalized predictions back to physical units (km)."""
        stats = self.stats[spacecraft_id]
        result = np.zeros_like(predictions)
        for i, col in enumerate(["x_gse", "y_gse", "z_gse"]):
            # Stats are stored under raw column names (x_gse, y_gse, z_gse)
            if col in stats["mean"]:
                result[..., i] = predictions[..., i] * stats["std"][col] + stats["mean"][col]
        return result

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Get position and velocity columns for normalization."""
        return [
            c for c in df.columns
            if c.startswith(("x_gse", "y_gse", "z_gse", "vx_gse", "vy_gse", "vz_gse"))
            and not c.endswith("_norm")
        ]


class SolarWindPreprocessor:
    """Preprocesses and aligns solar wind data with spacecraft positions."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.stats = {}

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize solar wind parameters and interpolate gaps."""
        df = df.copy()
        df = df.sort_values("time").reset_index(drop=True)

        param_cols = [c for c in df.columns if c != "time"]

        # Forward-fill small gaps (< 30 min), leave larger gaps as NaN
        df[param_cols] = df[param_cols].ffill(limit=30)

        # Normalize
        self.stats = {
            "mean": df[param_cols].mean().to_dict(),
            "std": df[param_cols].std().to_dict(),
        }

        for col in param_cols:
            std = self.stats["std"].get(col, 0)
            mean = self.stats["mean"].get(col, 0)
            if std and std > 0:
                df[f"{col}_norm"] = (df[col] - mean) / std
            else:
                df[f"{col}_norm"] = 0.0

        return df

    def align_with_positions(
        self,
        solar_df: pd.DataFrame,
        orbit_df: pd.DataFrame,
        propagation_delay_minutes: int = 45,
    ) -> pd.DataFrame:
        """Align solar wind data with spacecraft positions, accounting for L1 delay.

        The solar wind is measured at L1 (~1.5M km from Earth). It takes ~30-60 min
        for the solar wind to travel from L1 to Earth's magnetosphere.

        Args:
            solar_df: Preprocessed solar wind data
            orbit_df: Preprocessed orbit data
            propagation_delay_minutes: L1-to-Earth delay (default 45 min)
        """
        solar = solar_df.copy()

        # Shift solar wind timestamps forward by propagation delay
        solar["time"] = solar["time"] + pd.Timedelta(minutes=propagation_delay_minutes)

        # Ensure matching datetime precision/timezone for merge
        orbit_sorted = orbit_df.sort_values("time").copy()
        solar_sorted = solar.sort_values("time").copy()
        orbit_sorted["time"] = pd.to_datetime(orbit_sorted["time"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
        solar_sorted["time"] = pd.to_datetime(solar_sorted["time"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")

        # Merge on nearest timestamp (within 5 min tolerance)
        merged = pd.merge_asof(
            orbit_sorted,
            solar_sorted,
            on="time",
            tolerance=pd.Timedelta(minutes=5),
            direction="nearest",
        )

        return merged
