# Session 1: Models & Data — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce SGP4 baselines, storm-conditioned evaluation, shorter-horizon models, and ensemble results — all numbers that Sessions 2-4 depend on.

**Architecture:** New evaluation scripts load existing checkpoints and data, compute MAE under different conditions, and write results as JSON. SGP4 baseline uses a fresh implementation with real TLEs via `sgp4` library. Normalization stats are regenerated from training data and persisted alongside results.

**Tech Stack:** Python, PyTorch, sgp4, numpy, pandas, huggingface_hub. RunPod GPU for retraining (1c). Space-Track.org API for TLE download.

---

## CRITICAL: Pre-Implementation Fixes

**Apply these corrections everywhere in this plan before executing any step:**

### Fix 1: `OrbitPreprocessor.preprocess()` returns a single DataFrame
The API returns only a `pd.DataFrame`. Stats are stored on `self.stats`. Every occurrence of:
```python
processed, stats = proc.preprocess(df, spacecraft_id)
```
must be changed to:
```python
processed = proc.preprocess(df, spacecraft_id)
stats = proc.stats
```
Same for `SolarWindPreprocessor.preprocess()` — returns a DataFrame, stats on `self.stats`.

### Fix 2: Existing checkpoints use hidden_dim=64, NOT 128
The checkpoints in `checkpoints/` were saved with smaller architectures. The correct dimensions are:
- **LSTM**: `hidden_dim=64, num_layers=2`
- **Transformer**: `d_model=64, nhead=4, num_layers=2, dim_feedforward=128`
- **Multi-modal**: Check checkpoint — may use `hidden_dim=128, num_layers=3` (different training script)

Every `load_model()` call must use these dimensions. The plan's `load_model()` function should probe the checkpoint first:
```python
# Detect hidden_dim from checkpoint
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state = ckpt["model_state_dict"]
# LSTM: check lstm.weight_ih_l0 shape → (4*hidden_dim, input_dim)
if "lstm.weight_ih_l0" in state:
    hidden_dim = state["lstm.weight_ih_l0"].shape[0] // 4
elif "orbit_enc.weight_ih_l0" in state:
    hidden_dim = state["orbit_enc.weight_ih_l0"].shape[0] // 4
```

### Fix 3: `train_gpu.py` needs `--horizon-hours` argument
Before running Steps 27-30, add to `train_gpu.py`'s argparse section:
```python
parser.add_argument("--horizon-hours", type=int, default=6)
```
And update checkpoint naming to include horizon: `f"{model}_{sc}_{horizon}h_best.pt"`

### Fix 4: Ensemble must use real solar wind data, not zeros
Replace `solar_zeros = torch.zeros(...)` in `eval_ensemble.py` with actual solar wind windows from the same alignment logic used in `eval_storm_conditioned.py`. Import and reuse the solar wind preparation code.

### Fix 5: TLE SGP4 comparison is radial-only
The `compute_tle_mae()` compares radial distances (frame-invariant), not full 3D positions. This produces artificially low MAE. Results JSON should note `"note": "radial_distance_only"` and values should NOT be directly compared to ML model 3D MAE in the paper tables.

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `scripts/eval_sgp4.py` | Keplerian + TLE-based SGP4 baseline assessment |
| `scripts/eval_storm_conditioned.py` | Storm-conditioned MAE for all models + SGP4 |
| `scripts/eval_ensemble.py` | LSTM + multi-modal ensemble averaging |
| `scripts/download_tles.py` | Bulk TLE download from Space-Track.org |
| `results/sgp4_baselines.json` | SGP4 MAE per spacecraft |
| `results/storm_conditioned_mae.json` | MAE per model/spacecraft/condition |
| `results/horizon_comparison.json` | 1h/3h/6h MAE for LSTM + multi-modal on ISS |
| `results/norm_stats/` | Persisted normalization stats per spacecraft |
| `tests/test_preprocessing.py` | Unit tests for normalization and windowing |
| `tests/test_sgp4_baseline.py` | Unit tests for Keplerian propagation |

### Modified Files
| File | Changes |
|------|---------|
| `scripts/eval_checkpoints.py` | Add norm stats saving, fix hidden_dim mismatch |
| `src/models/baseline_sgp4.py` | Keep `simple_kepler_propagate`, add TLE loading helpers |

### Existing Files (read-only reference)
| File | Used For |
|------|----------|
| `scripts/train_gpu.py` | Model class definitions, training hyperparameters |
| `src/data/preprocessing.py` | `OrbitPreprocessor`, normalization, windowing |
| `src/data/solar_wind.py` | `SolarWindClient`, Kp alignment |
| `config.yaml` | Spacecraft IDs, split ratios, hyperparameters |

---

## Chunk 1: Normalization Stats & Test Infrastructure

### Task 1: Persist Normalization Stats

The biggest risk in this session is evaluating checkpoints with wrong normalization stats. Currently stats are computed on-the-fly and stored in memory. We need to persist them so all scripts use identical stats.

**Files:**
- Create: `results/norm_stats/` directory
- Modify: `scripts/eval_checkpoints.py:38-45` (add stats saving)
- Create: `tests/test_preprocessing.py`

- [ ] **Step 1: Create results directories**

```bash
mkdir -p results/norm_stats tests
touch tests/__init__.py
```

- [ ] **Step 2: Write failing test for stats persistence**

Create `tests/test_preprocessing.py`:

```python
"""Tests for normalization stats persistence."""
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path


def make_orbit_df(n=500):
    """Create a minimal orbit DataFrame for testing."""
    times = pd.date_range("2024-01-01", periods=n, freq="1min")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "time": times,
        "x_gse": rng.normal(0, 6000, n),
        "y_gse": rng.normal(0, 6000, n),
        "z_gse": rng.normal(0, 6000, n),
    })


def test_stats_save_and_load():
    """Stats saved to JSON can be loaded and reproduce identical normalization."""
    from src.data.preprocessing import OrbitPreprocessor

    df = make_orbit_df(1000)
    proc = OrbitPreprocessor()
    processed, stats = proc.preprocess(df, "test_sc")

    # Save stats
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(stats, f)
        stats_path = f.name

    # Load stats
    with open(stats_path) as f:
        loaded_stats = json.load(f)

    # Verify round-trip
    for col in ["x_gse", "y_gse", "z_gse", "vx_gse", "vy_gse", "vz_gse"]:
        assert abs(stats["test_sc"]["mean"][col] - loaded_stats["test_sc"]["mean"][col]) < 1e-10
        assert abs(stats["test_sc"]["std"][col] - loaded_stats["test_sc"]["std"][col]) < 1e-10

    Path(stats_path).unlink()


def test_denormalize_roundtrip():
    """Normalizing then denormalizing recovers original values."""
    from src.data.preprocessing import OrbitPreprocessor

    df = make_orbit_df(1000)
    proc = OrbitPreprocessor()
    processed, stats = proc.preprocess(df, "test_sc")

    # Get normalized position columns
    norm_cols = ["x_gse_norm", "y_gse_norm", "z_gse_norm"]
    norm_vals = processed[norm_cols].values[:5]

    # Manually denormalize
    denorm = np.zeros_like(norm_vals)
    for i, col in enumerate(["x_gse", "y_gse", "z_gse"]):
        denorm[:, i] = norm_vals[:, i] * stats["test_sc"]["std"][col] + stats["test_sc"]["mean"][col]

    # Compare to original
    orig = processed[["x_gse", "y_gse", "z_gse"]].values[:5]
    np.testing.assert_allclose(denorm, orig, rtol=1e-6)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /var/www/orbit && python -m pytest tests/test_preprocessing.py -v
```

Expected: FAIL — `OrbitPreprocessor.preprocess()` may not return stats in the expected format, or import path may need adjustment.

- [ ] **Step 4: Fix any import/API issues and verify tests pass**

If `OrbitPreprocessor.preprocess()` doesn't return `(df, stats)` tuple, check the actual API in `src/data/preprocessing.py:28-78` and adjust the test to match. The key is that stats contain `mean` and `std` dicts per spacecraft.

Run: `python -m pytest tests/test_preprocessing.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Add stats-saving to eval_checkpoints.py**

Add to `scripts/eval_checkpoints.py` after stats are computed (around line 45):

```python
import json
from pathlib import Path

# After preprocessing, save stats for this spacecraft
stats_dir = Path("results/norm_stats")
stats_dir.mkdir(parents=True, exist_ok=True)
stats_path = stats_dir / f"{spacecraft}_norm_stats.json"
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)
log.info(f"Saved normalization stats to {stats_path}")
```

- [ ] **Step 6: Run eval_checkpoints.py for ISS to generate and save stats**

```bash
cd /var/www/orbit && python scripts/eval_checkpoints.py --spacecraft iss
```

Expected: Stats saved to `results/norm_stats/iss_norm_stats.json`. Check file exists and contains `mean`/`std` dicts.

- [ ] **Step 7: Commit**

```bash
git add tests/ results/norm_stats/ scripts/eval_checkpoints.py
git commit -m "feat: persist normalization stats and add preprocessing tests"
```

---

### Task 2: Keplerian Baseline (Two-Body Propagation)

This is the simpler SGP4 baseline — pure physics with no TLE data needed. Uses Velocity Verlet integration to propagate orbits using Newtonian gravity.

**Files:**
- Create: `scripts/eval_sgp4.py`
- Create: `tests/test_sgp4_baseline.py`
- Read: `src/models/baseline_sgp4.py` (reuse `simple_kepler_propagate`)

- [ ] **Step 8: Write failing test for Keplerian baseline**

Create `tests/test_sgp4_baseline.py`:

```python
"""Tests for SGP4 and Keplerian baseline."""
import numpy as np


def test_kepler_propagate_circular_orbit():
    """A circular orbit should return to roughly the same position after one period."""
    from src.models.baseline_sgp4 import SGP4Baseline

    # ISS-like circular orbit: ~408 km altitude, ~6778 km radius
    r = 6778.0  # km
    mu = 398600.4418  # km^3/s^2
    v_circular = np.sqrt(mu / r)  # ~7.67 km/s

    pos = np.array([r, 0.0, 0.0])
    vel = np.array([0.0, v_circular, 0.0])

    period = 2 * np.pi * np.sqrt(r**3 / mu)  # ~5560 seconds (~92 min)
    dt = 60.0  # 1-minute steps
    n_steps = int(period / dt)

    result = SGP4Baseline.simple_kepler_propagate(pos, vel, dt, n_steps)

    # After one full period, should be close to start
    final_pos = result[-1]
    distance_from_start = np.linalg.norm(final_pos - pos)

    # Velocity Verlet is accurate — expect <10 km error after one orbit
    assert distance_from_start < 10.0, f"Expected <10 km, got {distance_from_start:.1f} km"


def test_kepler_propagate_output_shape():
    """Output shape should be (n_steps, 3)."""
    from src.models.baseline_sgp4 import SGP4Baseline

    pos = np.array([6778.0, 0.0, 0.0])
    vel = np.array([0.0, 7.67, 0.0])

    result = SGP4Baseline.simple_kepler_propagate(pos, vel, 60.0, 360)
    assert result.shape == (360, 3)


def test_kepler_mae_computation():
    """Same initial conditions should produce near-zero MAE."""
    from src.models.baseline_sgp4 import SGP4Baseline

    r = 6778.0
    mu = 398600.4418
    v = np.sqrt(mu / r)
    pos = np.array([r, 0.0, 0.0])
    vel = np.array([0.0, v, 0.0])

    true_trajectory = SGP4Baseline.simple_kepler_propagate(pos, vel, 60.0, 360)
    pred_trajectory = SGP4Baseline.simple_kepler_propagate(pos, vel, 60.0, 360)

    distances = np.sqrt(np.sum((pred_trajectory - true_trajectory)**2, axis=-1))
    mae = np.mean(distances)

    assert mae < 0.01, f"Expected ~0 MAE, got {mae:.4f}"
```

- [ ] **Step 9: Run test to verify it passes** (these test existing code)

```bash
cd /var/www/orbit && python -m pytest tests/test_sgp4_baseline.py -v
```

Expected: 3 tests PASS (we're testing existing `simple_kepler_propagate`)

- [ ] **Step 10: Write the Keplerian assessment script**

Create `scripts/eval_sgp4.py`:

```python
#!/usr/bin/env python3
"""Assess SGP4 and Keplerian baselines against ML model test sets.

Usage:
    python scripts/eval_sgp4.py --mode kepler --spacecraft iss
    python scripts/eval_sgp4.py --mode kepler --spacecraft all
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.baseline_sgp4 import SGP4Baseline
from src.data.preprocessing import OrbitPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sgp4-baseline")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def compute_kepler_mae(spacecraft_id: str, config: dict) -> float:
    """Compute Keplerian (two-body) baseline MAE on test set.

    For each test window:
    1. Take the last position and velocity from the input window
    2. Propagate forward using Velocity Verlet for horizon_steps
    3. Compare predicted positions to ground truth
    4. Return MAE in km
    """
    log.info(f"Computing Keplerian baseline for {spacecraft_id}")

    # Load and preprocess data
    proc = OrbitPreprocessor()
    raw_path = Path(f"data/raw/{spacecraft_id}_2023-01-01_2025-12-31.parquet")
    if not raw_path.exists():
        log.error(f"Data not found: {raw_path}")
        return float("nan")

    df = pd.read_parquet(raw_path)
    processed, stats = proc.preprocess(df, spacecraft_id)

    # Save stats for other scripts
    stats_dir = RESULTS_DIR / "norm_stats"
    stats_dir.mkdir(exist_ok=True)
    with open(stats_dir / f"{spacecraft_id}_norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Create windows at native resolution (subsample=1 for position/velocity)
    input_hours = config["model"]["input_hours"]
    horizon_hours = 6
    stride_hours = 6
    time_res = config["model"]["time_resolution_minutes"]
    input_steps = (input_hours * 60) // time_res
    horizon_steps = (horizon_hours * 60) // time_res
    stride_steps = (stride_hours * 60) // time_res

    # We need raw (un-normalized) position and velocity for Kepler propagation
    pos_cols = ["x_gse", "y_gse", "z_gse"]
    vel_cols = ["vx_gse", "vy_gse", "vz_gse"]
    all_cols = pos_cols + vel_cols

    # Build windows manually from raw values (not normalized)
    inputs_raw = []
    targets_raw = []

    for _, seg in processed.groupby("segment_id"):
        if len(seg) < input_steps + horizon_steps:
            continue
        feats = seg[all_cols].values
        tgts = seg[pos_cols].values
        for i in range(0, len(seg) - input_steps - horizon_steps, stride_steps):
            inputs_raw.append(feats[i:i + input_steps])
            targets_raw.append(tgts[i + input_steps:i + input_steps + horizon_steps])

    inputs_raw = np.array(inputs_raw, dtype=np.float64)
    targets_raw = np.array(targets_raw, dtype=np.float64)

    # Chronological test split (last 15%)
    n = len(inputs_raw)
    test_start = int(0.85 * n)
    test_inputs = inputs_raw[test_start:]
    test_targets = targets_raw[test_start:]

    log.info(f"Test set: {len(test_inputs)} windows")

    # Propagate from last position/velocity in each input window
    dt_seconds = time_res * 60.0  # 1 min = 60 seconds
    all_distances = []

    for i in range(len(test_inputs)):
        last_pos = test_inputs[i, -1, :3]   # Last position (km)
        last_vel = test_inputs[i, -1, 3:6]  # Last velocity (km/s)

        pred = SGP4Baseline.simple_kepler_propagate(
            last_pos, last_vel, dt_seconds, horizon_steps
        )

        distances = np.sqrt(np.sum((pred - test_targets[i])**2, axis=-1))
        all_distances.append(distances)

    all_distances = np.concatenate(all_distances)
    mae = float(np.mean(all_distances))
    rmse = float(np.sqrt(np.mean(all_distances**2)))

    log.info(f"{spacecraft_id} Keplerian: MAE={mae:.1f} km, RMSE={rmse:.1f} km")
    return mae


def main():
    parser = argparse.ArgumentParser(description="Assess SGP4/Keplerian baselines")
    parser.add_argument("--mode", choices=["kepler", "tle", "both"], default="kepler")
    parser.add_argument("--spacecraft", default="all")
    args = parser.parse_args()

    config = load_config()
    spacecraft_list = (
        list(config["spacecraft"].keys())
        if args.spacecraft == "all"
        else [args.spacecraft]
    )

    results = {}

    if args.mode in ("kepler", "both"):
        results["keplerian"] = {}
        for sc in spacecraft_list:
            results["keplerian"][sc] = compute_kepler_mae(sc, config)

    # TLE mode will be added in Task 3

    # Save results
    out_path = RESULTS_DIR / "sgp4_baselines.json"
    # Merge with existing results if file exists
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 11: Run Keplerian baseline on ISS**

```bash
cd /var/www/orbit && python scripts/eval_sgp4.py --mode kepler --spacecraft iss
```

Expected: Outputs MAE in km for ISS. Should be in the 200-800 km range for 6h prediction (two-body doesn't model drag, J2, etc.).

- [ ] **Step 12: Run on all spacecraft**

```bash
cd /var/www/orbit && python scripts/eval_sgp4.py --mode kepler --spacecraft all
```

Expected: Results for ISS, DSCOVR, MMS-1 saved to `results/sgp4_baselines.json`.

- [ ] **Step 13: Commit**

```bash
git add scripts/eval_sgp4.py tests/test_sgp4_baseline.py results/sgp4_baselines.json
git commit -m "feat: add Keplerian two-body baseline"
```

---

## Chunk 2: TLE-Based SGP4 Baseline

### Task 3: Download TLEs from Space-Track.org

**Files:**
- Create: `scripts/download_tles.py`
- Create: `data/tles/` directory

**Prerequisites:** Space-Track.org account (free). Set credentials as environment variables:
```bash
export SPACETRACK_USER="your_email"
export SPACETRACK_PASS="your_password"
```

- [ ] **Step 14: Create TLE download script**

Create `scripts/download_tles.py`:

```python
#!/usr/bin/env python3
"""Download TLEs from Space-Track.org for all tracked spacecraft.

Usage:
    export SPACETRACK_USER="email" SPACETRACK_PASS="password"
    python scripts/download_tles.py
"""
import json
import logging
import os
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tle-download")

# NORAD catalog IDs
NORAD_IDS = {
    "iss": 25544,
    "dscovr": 40390,
    "mms1": 40482,
}

TLE_DIR = Path("data/tles")
TLE_DIR.mkdir(parents=True, exist_ok=True)

SPACETRACK_BASE = "https://www.space-track.org"
LOGIN_URL = f"{SPACETRACK_BASE}/ajaxauth/login"
QUERY_URL = f"{SPACETRACK_BASE}/basicspacedata/query"


def download_tles():
    user = os.environ.get("SPACETRACK_USER")
    passwd = os.environ.get("SPACETRACK_PASS")
    if not user or not passwd:
        log.error("Set SPACETRACK_USER and SPACETRACK_PASS environment variables")
        return

    session = requests.Session()

    # Login
    resp = session.post(LOGIN_URL, data={"identity": user, "password": passwd})
    if resp.status_code != 200:
        log.error(f"Login failed: {resp.status_code}")
        return
    log.info("Logged in to Space-Track.org")

    for name, norad_id in NORAD_IDS.items():
        log.info(f"Downloading TLEs for {name} (NORAD {norad_id})")

        # Bulk query: all TLEs in date range, ordered by epoch
        url = (
            f"{QUERY_URL}/class/gp_history/NORAD_CAT_ID/{norad_id}"
            f"/EPOCH/2023-01-01--2025-12-31/orderby/EPOCH asc/format/tle"
        )

        resp = session.get(url)
        if resp.status_code != 200:
            log.error(f"Query failed for {name}: {resp.status_code}")
            continue

        tle_text = resp.text.strip()
        if not tle_text:
            log.warning(f"No TLEs returned for {name}")
            continue

        # Parse into list of (line1, line2) tuples
        lines = tle_text.strip().split("\n")
        tle_pairs = []
        for i in range(0, len(lines) - 1, 2):
            line1 = lines[i].strip()
            line2 = lines[i + 1].strip()
            if line1.startswith("1 ") and line2.startswith("2 "):
                tle_pairs.append({"line1": line1, "line2": line2})

        out_path = TLE_DIR / f"{name}_tles.json"
        with open(out_path, "w") as f:
            json.dump(tle_pairs, f)

        log.info(f"Saved {len(tle_pairs)} TLEs to {out_path}")

        # Respect rate limits
        time.sleep(5)

    session.close()
    log.info("Done")


if __name__ == "__main__":
    download_tles()
```

- [ ] **Step 15: Run TLE download** (requires Space-Track credentials)

```bash
export SPACETRACK_USER="your_email"
export SPACETRACK_PASS="your_password"
cd /var/www/orbit && python scripts/download_tles.py
```

Expected: JSON files in `data/tles/` with TLE pairs for each spacecraft.

- [ ] **Step 16: Commit TLE download script** (not the TLE data — it's large)

```bash
git add scripts/download_tles.py
echo "data/tles/" >> .gitignore
git add .gitignore
git commit -m "feat: add Space-Track.org TLE bulk download script"
```

---

### Task 4: TLE-Based SGP4 Assessment

**Files:**
- Modify: `scripts/eval_sgp4.py` (add `compute_tle_mae` function)
- Modify: `tests/test_sgp4_baseline.py` (add TLE loading test)

- [ ] **Step 17: Write test for TLE loading**

Add to `tests/test_sgp4_baseline.py`:

```python
def test_tle_loading_and_propagation():
    """Load a real TLE and propagate it forward."""
    from sgp4.api import Satrec, jday

    # ISS TLE from a known epoch (example)
    line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005"
    line2 = "2 25544  51.6400 208.0000 0007417  30.0000 330.0000 15.49000000000000"

    sat = Satrec.twoline2rv(line1, line2)
    jd, fr = jday(2024, 1, 1, 12, 0, 0)

    e, r, v = sat.sgp4(jd, fr)

    assert e == 0, f"SGP4 error code: {e}"
    assert len(r) == 3  # (x, y, z) in km
    assert len(v) == 3  # (vx, vy, vz) in km/s

    # ISS should be within ~7000 km of Earth center
    dist = np.sqrt(sum(x**2 for x in r))
    assert 6300 < dist < 7200, f"ISS distance from Earth: {dist:.0f} km"
```

- [ ] **Step 18: Run test to verify**

```bash
cd /var/www/orbit && python -m pytest tests/test_sgp4_baseline.py::test_tle_loading_and_propagation -v
```

Expected: PASS (sgp4 library is in requirements.txt)

- [ ] **Step 19: Add TLE function to eval_sgp4.py**

Add `compute_tle_mae()` function to `scripts/eval_sgp4.py` (after `compute_kepler_mae`):

```python
def compute_tle_mae(spacecraft_id: str, config: dict) -> float:
    """Compute TLE-based SGP4 baseline MAE on test set.

    For each test window:
    1. Find the closest TLE to the window start time
    2. Propagate SGP4 forward for horizon_steps
    3. Compare predicted positions (TEME frame) to ground truth (GSE frame)
    4. Return MAE in km

    Note: SGP4 outputs in TEME frame while our data is GSE. For a fair
    comparison we compare distances from Earth center, which are frame-invariant
    for position magnitude. For full 3D comparison, a TEME-to-GSE rotation is needed.
    """
    from sgp4.api import Satrec, jday
    from datetime import datetime, timedelta

    tle_path = Path(f"data/tles/{spacecraft_id}_tles.json")
    if not tle_path.exists():
        log.error(f"TLE data not found: {tle_path}. Run scripts/download_tles.py first.")
        return float("nan")

    with open(tle_path) as f:
        tle_pairs = json.load(f)

    log.info(f"Loaded {len(tle_pairs)} TLEs for {spacecraft_id}")

    # Parse all TLEs and extract epochs
    satellites = []
    for pair in tle_pairs:
        try:
            sat = Satrec.twoline2rv(pair["line1"], pair["line2"])
            epoch_jd = sat.jdsatepoch + sat.jdsatepochF
            satellites.append((epoch_jd, sat))
        except Exception:
            continue

    satellites.sort(key=lambda x: x[0])
    tle_epochs = np.array([s[0] for s in satellites])
    log.info(f"Parsed {len(satellites)} valid TLEs")

    # Load and preprocess orbit data
    proc = OrbitPreprocessor()
    raw_path = Path(f"data/raw/{spacecraft_id}_2023-01-01_2025-12-31.parquet")
    df = pd.read_parquet(raw_path)
    processed, stats = proc.preprocess(df, spacecraft_id)

    # Build test windows
    input_hours = config["model"]["input_hours"]
    time_res = config["model"]["time_resolution_minutes"]
    input_steps = (input_hours * 60) // time_res
    horizon_steps = (6 * 60) // time_res
    stride_steps = (6 * 60) // time_res

    pos_cols = ["x_gse", "y_gse", "z_gse"]
    windows_pos = []
    windows_time = []
    for _, seg in processed.groupby("segment_id"):
        if len(seg) < input_steps + horizon_steps:
            continue
        tgts = seg[pos_cols].values
        times = seg["time"].values
        for i in range(0, len(seg) - input_steps - horizon_steps, stride_steps):
            windows_pos.append(tgts[i + input_steps:i + input_steps + horizon_steps])
            windows_time.append(times[i + input_steps])

    windows_pos = np.array(windows_pos, dtype=np.float64)
    windows_time = np.array(windows_time)

    # Test split
    n = len(windows_pos)
    test_start = int(0.85 * n)
    test_targets = windows_pos[test_start:]
    test_times = windows_time[test_start:]

    log.info(f"Test set: {len(test_targets)} windows")

    # Run SGP4 predictions
    dt_minutes = time_res
    all_distances = []
    skipped = 0

    for i in range(len(test_targets)):
        start_dt = pd.Timestamp(test_times[i]).to_pydatetime()
        jd_start, fr_start = jday(
            start_dt.year, start_dt.month, start_dt.day,
            start_dt.hour, start_dt.minute, start_dt.second
        )
        jd_full = jd_start + fr_start

        # Find closest TLE
        idx = np.argmin(np.abs(tle_epochs - jd_full))
        sat = satellites[idx][1]

        # Propagate for each horizon step
        pred = np.zeros((horizon_steps, 3))
        valid = True
        for t in range(horizon_steps):
            minutes_ahead = (t + 1) * dt_minutes
            future_dt = start_dt + timedelta(minutes=minutes_ahead)
            jd_f, fr_f = jday(
                future_dt.year, future_dt.month, future_dt.day,
                future_dt.hour, future_dt.minute, future_dt.second
            )
            e, r, v = sat.sgp4(jd_f, fr_f)
            if e != 0:
                valid = False
                break
            pred[t] = r  # TEME frame (km)

        if not valid:
            skipped += 1
            continue

        # Compare radial distances (frame-invariant)
        pred_dist = np.sqrt(np.sum(pred**2, axis=-1))
        true_dist = np.sqrt(np.sum(test_targets[i]**2, axis=-1))
        distances = np.abs(pred_dist - true_dist)
        all_distances.append(distances)

    if skipped > 0:
        log.warning(f"Skipped {skipped}/{len(test_targets)} windows due to SGP4 errors")

    all_distances = np.concatenate(all_distances)
    mae = float(np.mean(all_distances))
    log.info(f"{spacecraft_id} TLE SGP4: MAE={mae:.1f} km (radial distance comparison)")
    return mae
```

Also update the `main()` function to handle TLE mode:

```python
    if args.mode in ("tle", "both"):
        results["sgp4_tle"] = {}
        for sc in spacecraft_list:
            results["sgp4_tle"][sc] = compute_tle_mae(sc, config)
```

- [ ] **Step 20: Run TLE assessment on ISS** (requires TLE data from Step 15)

```bash
cd /var/www/orbit && python scripts/eval_sgp4.py --mode tle --spacecraft iss
```

Expected: TLE-based SGP4 MAE for ISS. Should be lower than Keplerian since SGP4 models J2, drag, etc.

- [ ] **Step 21: Run all baselines**

```bash
cd /var/www/orbit && python scripts/eval_sgp4.py --mode both --spacecraft all
```

Expected: `results/sgp4_baselines.json` with both `keplerian` and `sgp4_tle` keys.

- [ ] **Step 22: Commit**

```bash
git add scripts/eval_sgp4.py scripts/download_tles.py tests/test_sgp4_baseline.py results/sgp4_baselines.json
git commit -m "feat: add TLE-based SGP4 baseline"
```

---

## Chunk 3: Storm-Conditioned Assessment

### Task 5: Storm-Conditioned MAE for All Models

This is the core research contribution — splitting results by geomagnetic conditions to show when solar wind data helps.

**Files:**
- Create: `scripts/eval_storm_conditioned.py`
- Read: `results/norm_stats/` (from Task 1)
- Read: `checkpoints/` (existing model checkpoints)
- Read: `data/raw/solar_wind_*.parquet`

- [ ] **Step 23: Create the storm-conditioned script**

Create `scripts/eval_storm_conditioned.py`:

```python
#!/usr/bin/env python3
"""Assess all model checkpoints under different geomagnetic conditions.

Splits test set by Kp index:
- Quiet:  Kp <= 3
- Active: 4 <= Kp <= 5
- Storm:  Kp >= 6

Usage:
    python scripts/eval_storm_conditioned.py --spacecraft iss
    python scripts/eval_storm_conditioned.py --spacecraft all
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.preprocessing import OrbitPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("storm-conditioned")

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")
DEVICE = torch.device("cpu")


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_model(model_type, checkpoint_path, input_dim=6, solar_dim=8, horizon=360):
    """Load a model checkpoint. Handles both single and dual-input models."""
    from scripts.train_gpu import OrbitLSTMDirect, OrbitTransformerDirect, SolarWindOrbitModel

    if model_type == "lstm":
        model = OrbitLSTMDirect(
            input_dim=input_dim, hidden_dim=128, num_layers=3,
            horizon=horizon, output_dim=3, dropout=0.1
        )
    elif model_type == "transformer":
        model = OrbitTransformerDirect(
            input_dim=input_dim, d_model=128, nhead=8, num_layers=4,
            dim_feedforward=512, horizon=horizon, output_dim=3, dropout=0.1
        )
    elif model_type == "multimodal":
        model = SolarWindOrbitModel(
            orbit_input_dim=input_dim, solar_input_dim=solar_dim,
            hidden_dim=128, num_layers=3, nhead=8,
            horizon=horizon, output_dim=3, dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    return model


def assign_kp_to_windows(window_start_times, solar_df):
    """Assign Kp value to each window using the preceding Kp report."""
    kp_df = solar_df[["time", "kp"]].dropna(subset=["kp"]).copy()
    kp_df = kp_df.sort_values("time").drop_duplicates(subset=["time"])

    windows_df = pd.DataFrame({"time": pd.to_datetime(window_start_times)})
    merged = pd.merge_asof(
        windows_df.sort_values("time"),
        kp_df.sort_values("time"),
        on="time",
        direction="backward"
    )
    return merged["kp"].values


def compute_conditioned_mae(
    model, model_type, test_inputs, test_targets, test_kp,
    stats, spacecraft_id, test_solar=None
):
    """Compute model MAE on test data split by Kp condition.

    Returns dict: {"all": mae, "quiet": mae, "active": mae, "storm": mae}
    """
    conditions = {
        "all": np.ones(len(test_kp), dtype=bool),
        "quiet": test_kp <= 3,
        "active": (test_kp >= 4) & (test_kp <= 5),
        "storm": test_kp >= 6,
    }

    model.eval()
    results = {}
    for cond_name, mask in conditions.items():
        n = mask.sum()
        if n == 0:
            log.warning(f"No samples for condition '{cond_name}'")
            results[cond_name] = None
            continue

        inputs_cond = test_inputs[mask]
        targets_cond = test_targets[mask]

        with torch.no_grad():
            x = torch.from_numpy(inputs_cond).float().to(DEVICE)
            if model_type == "multimodal" and test_solar is not None:
                solar_cond = test_solar[mask]
                sw = torch.from_numpy(solar_cond).float().to(DEVICE)
                preds = model(x, sw).cpu().numpy()
            else:
                preds = model(x).cpu().numpy()

        # Denormalize predictions and targets
        preds_km = np.zeros_like(preds)
        tgts_km = np.zeros_like(targets_cond)
        for i, col in enumerate(["x_gse", "y_gse", "z_gse"]):
            std = stats[spacecraft_id]["std"][col]
            mean = stats[spacecraft_id]["mean"][col]
            preds_km[..., i] = preds[..., i] * std + mean
            tgts_km[..., i] = targets_cond[..., i] * std + mean

        distances = np.sqrt(np.sum((preds_km - tgts_km)**2, axis=-1))
        mae = float(np.mean(distances))

        results[cond_name] = round(mae, 1)
        log.info(f"  {cond_name} (n={n}): MAE={mae:.1f} km")

    return results


def run_spacecraft(spacecraft_id, config):
    """Run storm-conditioned assessment for all models on one spacecraft."""
    log.info(f"=== Processing {spacecraft_id} ===")

    # Load orbit data and create test set
    proc = OrbitPreprocessor()
    raw_path = Path(f"data/raw/{spacecraft_id}_2023-01-01_2025-12-31.parquet")
    df = pd.read_parquet(raw_path)
    processed, stats = proc.preprocess(df, spacecraft_id)

    # Create normalized windows
    input_hours = config["model"]["input_hours"]
    time_res = config["model"]["time_resolution_minutes"]
    input_steps = (input_hours * 60) // time_res
    horizon_steps = (6 * 60) // time_res
    stride_steps = (6 * 60) // time_res

    norm_feat_cols = sorted([c for c in processed.columns if c.endswith("_norm")])
    norm_tgt_cols = ["x_gse_norm", "y_gse_norm", "z_gse_norm"]

    inputs, targets, window_times = [], [], []
    for _, seg in processed.groupby("segment_id"):
        if len(seg) < input_steps + horizon_steps:
            continue
        feats = seg[norm_feat_cols].values
        tgts = seg[norm_tgt_cols].values
        times = seg["time"].values
        for i in range(0, len(seg) - input_steps - horizon_steps, stride_steps):
            inputs.append(feats[i:i + input_steps])
            targets.append(tgts[i + input_steps:i + input_steps + horizon_steps])
            window_times.append(times[i + input_steps])

    inputs = np.array(inputs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    window_times = np.array(window_times)

    # Test split
    n = len(inputs)
    test_start = int(0.85 * n)
    test_inputs = inputs[test_start:]
    test_targets = targets[test_start:]
    test_times = window_times[test_start:]

    # Load solar wind data for Kp assignment
    sw_path = Path("data/raw/solar_wind_2023-01-01_2025-12-31.parquet")
    solar_df = pd.read_parquet(sw_path)
    test_kp = assign_kp_to_windows(test_times, solar_df)

    log.info(f"Test set: {len(test_inputs)} windows")
    log.info(f"Kp distribution: quiet={np.sum(test_kp <= 3)}, "
             f"active={np.sum((test_kp >= 4) & (test_kp <= 5))}, "
             f"storm={np.sum(test_kp >= 6)}")

    # Prepare solar wind windows for multi-modal
    test_solar = None
    try:
        from src.data.preprocessing import SolarWindPreprocessor
        sw_proc = SolarWindPreprocessor()
        solar_processed, solar_stats = sw_proc.preprocess(solar_df)
        aligned = sw_proc.align_with_positions(solar_processed, processed)

        solar_norm_cols = sorted([c for c in aligned.columns
                                  if c.endswith("_norm") and c.split("_norm")[0] in
                                  ["bx_gse", "by_gse", "bz_gse", "flow_speed",
                                   "proton_density", "kp", "dst", "ae"]])

        if solar_norm_cols:
            solar_inputs = []
            for _, seg in aligned.groupby("segment_id"):
                if len(seg) < input_steps + horizon_steps:
                    continue
                sw_feats = seg[solar_norm_cols].values
                for i in range(0, len(seg) - input_steps - horizon_steps, stride_steps):
                    solar_inputs.append(sw_feats[i:i + input_steps])

            if len(solar_inputs) == len(inputs):
                solar_inputs = np.array(solar_inputs, dtype=np.float32)
                test_solar = solar_inputs[test_start:]
            else:
                log.warning(f"Solar window count mismatch: {len(solar_inputs)} vs {len(inputs)}")
    except Exception as e:
        log.warning(f"Could not prepare solar wind windows: {e}")

    # Run each model
    model_results = {}
    for model_type in ["lstm", "transformer", "multimodal"]:
        ckpt_path = CHECKPOINT_DIR / f"{model_type}_{spacecraft_id}_6h_best.pt"
        if not ckpt_path.exists():
            log.warning(f"Checkpoint not found: {ckpt_path}")
            continue

        log.info(f"Running {model_type}")
        try:
            model = load_model(model_type, ckpt_path, input_dim=inputs.shape[-1])
            model_results[model_type] = compute_conditioned_mae(
                model, model_type, test_inputs, test_targets, test_kp,
                stats, spacecraft_id, test_solar=test_solar
            )
        except Exception as e:
            log.error(f"Failed on {model_type}: {e}")
            model_results[model_type] = {"all": None, "quiet": None, "active": None, "storm": None}

    return model_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacecraft", default="all")
    args = parser.parse_args()

    config = load_config()
    spacecraft_list = (
        list(config["spacecraft"].keys())
        if args.spacecraft == "all"
        else [args.spacecraft]
    )

    all_results = {}
    for sc in spacecraft_list:
        sc_results = run_spacecraft(sc, config)
        for model_type, cond_results in sc_results.items():
            if model_type not in all_results:
                all_results[model_type] = {}
            all_results[model_type][sc] = cond_results

    out_path = RESULTS_DIR / "storm_conditioned_mae.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 24: Run storm-conditioned assessment on ISS**

```bash
cd /var/www/orbit && python scripts/eval_storm_conditioned.py --spacecraft iss
```

Expected: MAE broken down by quiet/active/storm for LSTM, Transformer, Multi-modal.

- [ ] **Step 25: Run on all spacecraft**

```bash
cd /var/www/orbit && python scripts/eval_storm_conditioned.py --spacecraft all
```

Expected: Full `results/storm_conditioned_mae.json`.

- [ ] **Step 26: Commit**

```bash
git add scripts/eval_storm_conditioned.py results/storm_conditioned_mae.json
git commit -m "feat: add storm-conditioned model assessment"
```

---

## Chunk 4: Shorter Horizons & Ensemble

### Task 6: Retrain at 1h and 3h Horizons (RunPod)

This requires GPU access. Train LSTM and multi-modal on ISS at 1h (60 steps) and 3h (180 steps).

**Files:**
- Use: `scripts/train_gpu.py` (existing, with modified horizon args)
- Output: 4 new checkpoints in `checkpoints/`

**Note:** `train_gpu.py` will need a `--horizon-hours` argument added. If it doesn't exist yet, add it to the argparse section.

- [ ] **Step 27: Train on RunPod — 1h LSTM**

```bash
python scripts/train_gpu.py --model lstm --spacecraft iss --horizon-hours 1 --epochs 100
```

Expected: Checkpoint saved as `checkpoints/lstm_iss_1h_best.pt`

- [ ] **Step 28: Train on RunPod — 3h LSTM**

```bash
python scripts/train_gpu.py --model lstm --spacecraft iss --horizon-hours 3 --epochs 100
```

Expected: Checkpoint saved as `checkpoints/lstm_iss_3h_best.pt`

- [ ] **Step 29: Train on RunPod — 1h Multi-modal**

```bash
python scripts/train_gpu.py --model multimodal --spacecraft iss --horizon-hours 1 --epochs 100
```

Expected: Checkpoint saved as `checkpoints/multimodal_iss_1h_best.pt`

- [ ] **Step 30: Train on RunPod — 3h Multi-modal**

```bash
python scripts/train_gpu.py --model multimodal --spacecraft iss --horizon-hours 3 --epochs 100
```

Expected: Checkpoint saved as `checkpoints/multimodal_iss_3h_best.pt`

- [ ] **Step 31: Record results and save**

After training, record the MAE values and create `results/horizon_comparison.json`:

```bash
cd /var/www/orbit && python3 -c "
import json
# Fill in actual MAE values from training output
results = {
    'lstm_iss': {'1h': LSTM_1H_MAE, '3h': LSTM_3H_MAE, '6h': 126},
    'multimodal_iss': {'1h': MM_1H_MAE, '3h': MM_3H_MAE, '6h': 175}
}
with open('results/horizon_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved horizon comparison results')
"
```

- [ ] **Step 32: Push new checkpoints to HF**

```bash
cd /var/www/orbit && python3 -c "
from huggingface_hub import HfApi
api = HfApi()
for name in ['lstm_iss_1h_best.pt', 'lstm_iss_3h_best.pt', 'multimodal_iss_1h_best.pt', 'multimodal_iss_3h_best.pt']:
    api.upload_file(
        path_or_fileobj=f'checkpoints/{name}',
        path_in_repo=f'checkpoints/{name}',
        repo_id='datamatters24/orbital-chaos-predictor',
        repo_type='model',
    )
    print(f'Uploaded {name}')
"
```

- [ ] **Step 33: Commit results**

```bash
git add results/horizon_comparison.json
git commit -m "feat: add 1h and 3h horizon comparison results"
```

**Fallback:** If RunPod is unavailable, use CPU with `OrbitLSTMDirect`, `--subsample 4`, `--epochs 30`. Skip multi-modal horizons (GPU required).

---

### Task 7: Ensemble

Average LSTM and multi-modal predictions on the same test samples.

**Files:**
- Create: `scripts/eval_ensemble.py`

- [ ] **Step 34: Create ensemble script**

Create `scripts/eval_ensemble.py`:

```python
#!/usr/bin/env python3
"""Average LSTM + Multi-modal predictions for ensemble results.

Usage:
    python scripts/eval_ensemble.py --spacecraft iss
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ensemble")

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacecraft", default="iss")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    sc = args.spacecraft

    from scripts.eval_storm_conditioned import (
        load_model, assign_kp_to_windows
    )
    from src.data.preprocessing import OrbitPreprocessor

    # Load data and create test set
    proc = OrbitPreprocessor()
    raw_path = Path(f"data/raw/{sc}_2023-01-01_2025-12-31.parquet")
    df = pd.read_parquet(raw_path)
    processed, stats = proc.preprocess(df, sc)

    input_hours = config["model"]["input_hours"]
    time_res = config["model"]["time_resolution_minutes"]
    input_steps = (input_hours * 60) // time_res
    horizon_steps = (6 * 60) // time_res
    stride_steps = (6 * 60) // time_res

    norm_feat_cols = sorted([c for c in processed.columns if c.endswith("_norm")])
    norm_tgt_cols = ["x_gse_norm", "y_gse_norm", "z_gse_norm"]

    inputs, targets, window_times = [], [], []
    for _, seg in processed.groupby("segment_id"):
        if len(seg) < input_steps + horizon_steps:
            continue
        feats = seg[norm_feat_cols].values
        tgts = seg[norm_tgt_cols].values
        times = seg["time"].values
        for i in range(0, len(seg) - input_steps - horizon_steps, stride_steps):
            inputs.append(feats[i:i + input_steps])
            targets.append(tgts[i + input_steps:i + input_steps + horizon_steps])
            window_times.append(times[i + input_steps])

    inputs = np.array(inputs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    window_times = np.array(window_times)

    n = len(inputs)
    test_start = int(0.85 * n)
    test_inputs = inputs[test_start:]
    test_targets = targets[test_start:]
    test_times = window_times[test_start:]

    # Load Kp
    sw_path = Path("data/raw/solar_wind_2023-01-01_2025-12-31.parquet")
    solar_df = pd.read_parquet(sw_path)
    test_kp = assign_kp_to_windows(test_times, solar_df)

    # Load LSTM
    lstm = load_model("lstm", CHECKPOINT_DIR / f"lstm_{sc}_6h_best.pt", input_dim=inputs.shape[-1])
    lstm.eval()

    log.info("Running LSTM predictions...")
    with torch.no_grad():
        x = torch.from_numpy(test_inputs).float()
        lstm_preds = lstm(x).numpy()

    log.info("Running Multi-modal predictions...")
    try:
        mm = load_model("multimodal", CHECKPOINT_DIR / f"multimodal_{sc}_6h_best.pt",
                        input_dim=inputs.shape[-1])
        mm.eval()
        with torch.no_grad():
            # Multi-modal needs solar input — use zeros (gate should suppress)
            solar_zeros = torch.zeros(len(test_inputs), input_steps, 8)
            mm_preds = mm(x, solar_zeros).numpy()
    except Exception as e:
        log.error(f"Multi-modal failed: {e}. Using LSTM only.")
        mm_preds = lstm_preds

    # Ensemble: simple average
    ensemble_preds = (lstm_preds + mm_preds) / 2.0

    # Per-condition MAE
    conditions = {
        "all": np.ones(len(test_kp), dtype=bool),
        "quiet": test_kp <= 3,
        "active": (test_kp >= 4) & (test_kp <= 5),
        "storm": test_kp >= 6,
    }

    ensemble_results = {}
    for cond_name, mask in conditions.items():
        n_cond = mask.sum()
        if n_cond == 0:
            ensemble_results[cond_name] = None
            continue

        preds_cond = ensemble_preds[mask]
        tgts_cond = test_targets[mask]

        preds_km = np.zeros_like(preds_cond)
        tgts_km = np.zeros_like(tgts_cond)
        for i, col in enumerate(["x_gse", "y_gse", "z_gse"]):
            std = stats[sc]["std"][col]
            mean = stats[sc]["mean"][col]
            preds_km[..., i] = preds_cond[..., i] * std + mean
            tgts_km[..., i] = tgts_cond[..., i] * std + mean

        distances = np.sqrt(np.sum((preds_km - tgts_km)**2, axis=-1))
        mae = round(float(np.mean(distances)), 1)
        ensemble_results[cond_name] = mae
        log.info(f"Ensemble {cond_name} (n={n_cond}): MAE={mae:.1f} km")

    # Merge into storm_conditioned_mae.json
    storm_path = RESULTS_DIR / "storm_conditioned_mae.json"
    if storm_path.exists():
        with open(storm_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    if "ensemble" not in all_results:
        all_results["ensemble"] = {}
    all_results["ensemble"][sc] = ensemble_results

    with open(storm_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"Ensemble results saved to {storm_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 35: Run ensemble**

```bash
cd /var/www/orbit && python scripts/eval_ensemble.py --spacecraft iss
```

Expected: Ensemble MAE between LSTM and multi-modal (often better than both due to error decorrelation).

- [ ] **Step 36: Commit**

```bash
git add scripts/eval_ensemble.py results/storm_conditioned_mae.json
git commit -m "feat: add LSTM + multi-modal ensemble"
```

---

## Chunk 5: Final Verification & Push

### Task 8: Verify All Results and Push

- [ ] **Step 37: Verify all result files exist**

```bash
ls -la results/sgp4_baselines.json results/storm_conditioned_mae.json results/horizon_comparison.json results/norm_stats/
```

Expected: All 3 JSON files and norm_stats directory with per-spacecraft stats.

- [ ] **Step 38: Print summary of all results**

```bash
cd /var/www/orbit && python3 -c "
import json
for f in ['results/sgp4_baselines.json', 'results/storm_conditioned_mae.json', 'results/horizon_comparison.json']:
    print(f'\n=== {f} ===')
    with open(f) as fh:
        print(json.dumps(json.load(fh), indent=2))
"
```

- [ ] **Step 39: Push all results to git**

```bash
git add results/
git commit -m "feat: complete Session 1 - all model results"
git push origin main
```

- [ ] **Step 40: Upload result files to HF**

```bash
cd /var/www/orbit && python3 -c "
from huggingface_hub import HfApi
api = HfApi()
for f in ['sgp4_baselines.json', 'storm_conditioned_mae.json', 'horizon_comparison.json']:
    api.upload_file(
        path_or_fileobj=f'results/{f}',
        path_in_repo=f'results/{f}',
        repo_id='datamatters24/orbital-chaos-predictor',
        repo_type='model',
    )
    print(f'Uploaded {f}')
"
```

---

## Session 1 Completion Checklist

- [ ] `results/sgp4_baselines.json` — Keplerian + TLE MAE for 3 spacecraft
- [ ] `results/storm_conditioned_mae.json` — MAE per model/spacecraft/condition + ensemble
- [ ] `results/horizon_comparison.json` — 1h/3h/6h MAE for LSTM + multi-modal on ISS
- [ ] `results/norm_stats/` — Persisted normalization stats for all spacecraft
- [ ] 4 new checkpoints on HF (1h/3h horizons for LSTM/multi-modal)
- [ ] All scripts committed and pushed
- [ ] Tests passing: `python -m pytest tests/ -v`

**Session 2 can now begin** — it reads these JSON files for the toggle table and site updates.
