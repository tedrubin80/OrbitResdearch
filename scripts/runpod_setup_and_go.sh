#!/bin/bash
set -e
echo "=== ORBITAL CHAOS — RUNPOD EXPANDED FEATURES ==="
echo "Started: $(date)"

# Install deps
pip install pyarrow huggingface_hub pyyaml sscws astropy safetensors pandas -q

# Setup workspace
mkdir -p /workspace/OrbitResearch
cd /workspace/OrbitResearch

# Download code and checkpoints
echo ">>> Downloading code from HF..."
hf download datamatters24/orbital-chaos-predictor --repo-type model --local-dir .

# Download data (includes pre-expanded solar wind with 13 features)
echo ">>> Downloading data from HF..."
hf download datamatters24/orbital-chaos-nasa-ssc --repo-type dataset --local-dir data/raw
mv data/raw/data/*.parquet data/raw/ 2>/dev/null || true
mkdir -p results checkpoints

# Verify solar wind has expanded features
echo ">>> Verifying solar wind features..."
python -c "
import pandas as pd
df = pd.read_parquet('data/raw/solar_wind_2023-01-01_2025-12-31.parquet')
cols = sorted(df.columns.tolist())
print(f'Solar wind: {len(df)} rows, {len(cols)} columns: {cols}')
expected = ['al','au','clock_angle_sin','clock_angle_cos','dynamic_pressure']
found = [c for c in expected if c in cols]
missing = [c for c in expected if c not in cols]
if missing:
    print(f'MISSING expanded features: {missing}')
    print('Adding derived features from existing columns...')
    import numpy as np
    if 'by_gse' in cols and 'bz_gse' in cols and 'clock_angle_sin' not in cols:
        ca = np.arctan2(df['by_gse'], df['bz_gse'])
        df['clock_angle_sin'] = np.sin(ca)
        df['clock_angle_cos'] = np.cos(ca)
    if 'proton_density' in cols and 'flow_speed' in cols and 'dynamic_pressure' not in cols:
        df['dynamic_pressure'] = 1.6726e-6 * df['proton_density'] * df['flow_speed']**2
    df.to_parquet('data/raw/solar_wind_2023-01-01_2025-12-31.parquet', index=False)
    print(f'Updated: {len(df)} rows, {len(df.columns)} columns')
else:
    print(f'All expanded features present!')
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
