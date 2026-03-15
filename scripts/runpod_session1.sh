#!/bin/bash
set -e
echo "=== ORBITAL CHAOS — SESSION 1 ON RUNPOD ==="
echo "Started: $(date)"

cd /OrbitResearch

# Force refresh all eval scripts
echo ""
echo ">>> Refreshing scripts from HF..."
rm -f scripts/eval_sgp4.py scripts/eval_storm_conditioned.py scripts/eval_ensemble.py
hf download datamatters24/orbital-chaos-predictor scripts/eval_sgp4.py --repo-type model --local-dir . --force-download
hf download datamatters24/orbital-chaos-predictor scripts/eval_storm_conditioned.py --repo-type model --local-dir . --force-download
hf download datamatters24/orbital-chaos-predictor scripts/eval_ensemble.py --repo-type model --local-dir . --force-download
hf download datamatters24/orbital-chaos-predictor scripts/train_gpu.py --repo-type model --local-dir . --force-download

# Step 1: SGP4 baseline
echo ""
echo ">>> STEP 1: SGP4 Keplerian baseline..."
python scripts/eval_sgp4.py --spacecraft all
echo "--- SGP4 Results ---"
cat results/sgp4_baselines.json
echo ""

# Step 2: Storm-conditioned eval
echo ""
echo ">>> STEP 2: Storm-conditioned evaluation..."
python scripts/eval_storm_conditioned.py --spacecraft all
echo "--- Storm Results ---"
cat results/storm_conditioned_mae.json
echo ""

# Step 3: GPU training — shorter horizons
echo ""
echo ">>> STEP 3: Training LSTM 1h..."
python scripts/train_gpu.py --model lstm --spacecraft iss --horizon-hours 1 --no-push

echo ""
echo ">>> STEP 4: Training LSTM 3h..."
python scripts/train_gpu.py --model lstm --spacecraft iss --horizon-hours 3 --no-push

echo ""
echo ">>> STEP 5: Training Multi-modal 1h..."
python scripts/train_gpu.py --model multimodal --spacecraft iss --horizon-hours 1 --no-push

echo ""
echo ">>> STEP 6: Training Multi-modal 3h..."
python scripts/train_gpu.py --model multimodal --spacecraft iss --horizon-hours 3 --no-push

# Step 4: Ensemble
echo ""
echo ">>> STEP 7: Ensemble evaluation..."
python scripts/eval_ensemble.py --spacecraft iss

# Final results
echo ""
echo "============================================"
echo "=== ALL DONE ==="
echo "============================================"
echo "Finished: $(date)"
echo ""
echo "--- SGP4 Baselines ---"
cat results/sgp4_baselines.json
echo ""
echo "--- Storm-Conditioned MAE ---"
cat results/storm_conditioned_mae.json
echo ""
echo "--- Horizon Checkpoints ---"
ls -la checkpoints/*_1h_* checkpoints/*_3h_* 2>/dev/null || echo "No horizon checkpoints found"
echo ""
echo "COPY EVERYTHING ABOVE AND PASTE BACK TO CLAUDE"
