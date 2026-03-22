#!/bin/bash
set -e
echo "=== ORBITAL CHAOS — ONE-SHOT RUNPOD ==="
echo "Started: $(date)"

pip install pyarrow huggingface_hub pyyaml sscws astropy safetensors pandas cdflib xarray -q

mkdir -p /workspace/OrbitResearch
cd /workspace/OrbitResearch

echo ">>> Downloading code..."
hf download datamatters24/orbital-chaos-predictor --repo-type model --local-dir .

echo ">>> Downloading data..."
hf download datamatters24/orbital-chaos-nasa-ssc --repo-type dataset --local-dir data/raw
mv data/raw/data/*.parquet data/raw/ 2>/dev/null || true
mkdir -p results checkpoints

echo ">>> Checking solar wind features..."
python << 'PYEOF'
import pandas as pd, numpy as np
df = pd.read_parquet('data/raw/solar_wind_2023-01-01_2025-12-31.parquet')
print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
if "clock_angle_sin" not in df.columns and "by_gse" in df.columns:
    ca = np.arctan2(df["by_gse"], df["bz_gse"])
    df["clock_angle_sin"] = np.sin(ca)
    df["clock_angle_cos"] = np.cos(ca)
    print("  Added clock_angle_sin/cos")
if "dynamic_pressure" not in df.columns and "proton_density" in df.columns:
    df["dynamic_pressure"] = 1.6726e-6 * df["proton_density"] * df["flow_speed"]**2
    print("  Added dynamic_pressure")
df.to_parquet("data/raw/solar_wind_2023-01-01_2025-12-31.parquet", index=False)
cols = sorted([c for c in df.columns if c != "time"])
print(f"Final: {len(cols)} features: {cols}")
PYEOF

echo ">>> Fixing batch size for GPU memory..."
sed -i 's/batch_size=64/batch_size=8/g' scripts/train_expanded_features.py

echo ">>> Starting training..."
python scripts/train_expanded_features.py --skip-ablation 2>&1 | tee /tmp/output.log

echo ""
echo "=== DONE ==="
echo "Finished: $(date)"
cat results/storm_eval_expanded.csv 2>/dev/null
cat results/expanded_feature_summary.md 2>/dev/null
echo "COPY ABOVE AND PASTE BACK TO CLAUDE"
