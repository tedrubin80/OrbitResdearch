#!/bin/bash
set -e
echo "=== ORBITAL CHAOS — RUNPOD SETUP + EXPANDED FEATURES ==="
echo "Started: $(date)"

# Install deps
pip install pyarrow huggingface_hub pyyaml sscws astropy cdasws safetensors -q

# Setup workspace
mkdir -p /workspace/OrbitResearch
cd /workspace/OrbitResearch

# Download code and checkpoints
echo ">>> Downloading code from HF..."
hf download datamatters24/orbital-chaos-predictor --repo-type model --local-dir .

# Download data
echo ">>> Downloading data from HF..."
hf download datamatters24/orbital-chaos-nasa-ssc --repo-type dataset --local-dir data/raw
mv data/raw/data/*.parquet data/raw/ 2>/dev/null || true
mkdir -p results checkpoints

# Clear old solar wind cache and re-fetch from CDAWeb with expanded features (AL, AU)
echo ">>> Clearing cached solar wind..."
rm -f data/raw/solar_wind_2023-01-01_2025-12-31.parquet

echo ">>> Re-fetching solar wind from CDAWeb with expanded config (AL, AU indices)..."
pip install cdasws -q 2>/dev/null
python -c "
import sys; sys.path.insert(0, '.')
from src.data.solar_wind import SolarWindClient
client = SolarWindClient()
df = client.fetch_solar_wind('2023-01-01', '2025-12-31')
print(f'Solar wind: {len(df)} rows, columns: {sorted(df.columns.tolist())}')
"

echo ">>> Verifying expanded features..."
python -c "
import pandas as pd
df = pd.read_parquet('data/raw/solar_wind_2023-01-01_2025-12-31.parquet')
expected = ['bx_gse','by_gse','bz_gse','flow_speed','proton_density','kp','dst','ae','al','au','clock_angle_sin','clock_angle_cos','dynamic_pressure']
found = [c for c in expected if c in df.columns]
missing = [c for c in expected if c not in df.columns]
print(f'Found {len(found)}/13 features: {found}')
if missing: print(f'MISSING: {missing}')
"

# Run the expanded feature training + ablation
echo ""
echo ">>> Starting expanded feature training..."
python scripts/train_expanded_features.py

echo ""
echo "============================================"
echo "=== ALL DONE ==="
echo "============================================"
echo "Finished: $(date)"
echo ""
echo "--- Ablation Results ---"
cat results/ablation_results.csv 2>/dev/null || echo "No ablation results"
echo ""
echo "--- Storm Evaluation ---"
cat results/storm_eval_expanded.csv 2>/dev/null || echo "No storm results"
echo ""
echo "--- Summary ---"
cat results/expanded_feature_summary.md 2>/dev/null || echo "No summary"
echo ""
echo "COPY EVERYTHING ABOVE AND PASTE BACK TO CLAUDE"
