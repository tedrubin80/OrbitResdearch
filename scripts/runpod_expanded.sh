#!/bin/bash
set -e
echo "=== EXPANDED FEATURE TRAINING ==="
echo "Started: $(date)"

# Setup
pip install pyarrow huggingface_hub pyyaml sscws astropy cdasws safetensors -q
mkdir -p OrbitResearch && cd OrbitResearch

# Pull latest code and data
hf download datamatters24/orbital-chaos-predictor --repo-type model --local-dir . --force-download
hf download datamatters24/orbital-chaos-nasa-ssc --repo-type dataset --local-dir data/raw
mv data/raw/data/*.parquet data/raw/ 2>/dev/null || true
mkdir -p results

# Need to re-fetch solar wind with expanded features (AL, AU, derived)
# Delete cached solar wind to force re-fetch with new config
echo "Clearing cached solar wind data to fetch expanded features..."
rm -f data/raw/solar_wind_2023-01-01_2025-12-31.parquet

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
echo "COPY EVERYTHING ABOVE AND PASTE BACK"
