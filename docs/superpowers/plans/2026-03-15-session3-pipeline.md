# Session 3: Pipeline & Infrastructure — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automate hourly ISS predictions, weekly model retraining, and health monitoring so the site runs unattended.

**Architecture:** Three new cron scripts. `cron_predict.py` loads the LSTM checkpoint, fetches live ISS data from NASA SSC, runs CPU inference, and writes `public/predictions.json`. `cron_retrain.py` retrains weekly with new data and pushes improved checkpoints to HF. `health_check.py` validates all services every 15 minutes.

**Tech Stack:** Python, PyTorch (CPU-only), astropy (coordinate transforms), NASA SSC API, cron.

**Prerequisites:** Sessions 1-2 completed. Normalization stats saved in `results/norm_stats/`. PyTorch CPU installed on server.

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `scripts/cron_predict.py` | Hourly: fetch ISS data, run LSTM, write predictions.json |
| `scripts/cron_retrain.py` | Weekly: retrain LSTM on latest data, push to HF if improved |
| `scripts/health_check.py` | Every 15 min: verify site, APIs, predictions freshness, disk |

### Modified Files
| File | Changes |
|------|---------|
| `scripts/cron_setup.sh` | Add 3 new cron entries |

---

## CRITICAL: Server Setup

- [ ] **Step 0a: Install PyTorch CPU**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -c "import torch; print(torch.__version__)"
```

- [ ] **Step 0b: Verify astropy**

```bash
python3 -c "import astropy; print(astropy.__version__)" || pip install astropy
```

- [ ] **Step 0c: Verify checkpoint and stats**

```bash
ls -la checkpoints/lstm_iss_6h_best.pt results/norm_stats/iss_norm_stats.json
```

---

## Chunk 1: Prediction Cron

### Task 1: Hourly ISS Prediction Script

**Files:** Create `scripts/cron_predict.py`

The script follows this flow:
1. Load saved normalization stats from `results/norm_stats/iss_norm_stats.json`
2. Fetch last 24h of ISS positions from NASA SSC API via `src/data/ssc_client.py`
3. Preprocess: derive velocity, normalize with **training stats** (not recomputed)
4. Detect model architecture from checkpoint weights (hidden_dim, num_layers)
5. Run LSTM inference on CPU (<5 seconds)
6. Denormalize GSE positions, convert to lat/lon/alt via `src/utils/coords.py`
7. Decimate to ~30 points, validate ranges (lat -90..90, alt 300..500 for ISS)
8. Write `public/predictions.json` with ISO timestamp

**Key implementation details:**
- Uses `OrbitPreprocessor.preprocess()` which returns a DataFrame (stats on `self.stats`)
- Must override `proc.stats` with saved training stats before re-normalizing
- Architecture detection: `state["lstm.weight_ih_l0"].shape[0] // 4` gives hidden_dim
- Returns exit code 0 on success, 1 on failure (for cron monitoring)

- [ ] **Step 1: Write `scripts/cron_predict.py`** — full script implementing the flow above

- [ ] **Step 2: Test locally**

```bash
cd /var/www/orbit && python3 scripts/cron_predict.py
cat public/predictions.json | python3 -m json.tool | head -15
```

Expected: JSON with `generated_at`, `model`, `path` array of lat/lng/alt objects.

- [ ] **Step 3: Verify predictions are valid**

```bash
python3 -c "
import json
d = json.load(open('public/predictions.json'))
print(f'Generated: {d[\"generated_at\"]}')
print(f'Points: {len(d[\"path\"])}')
p = d['path'][0]
print(f'First point: lat={p[\"lat\"]}, lng={p[\"lng\"]}, alt={p[\"alt\"]}')
assert -90 <= p['lat'] <= 90
assert -180 <= p['lng'] <= 180
assert 300 <= p['alt'] <= 500
print('Validation passed')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/cron_predict.py && git commit -m "feat: add hourly ISS prediction cron script"
```

---

## Chunk 2: Weekly Retrain

### Task 2: Auto-Retrain Script

**Files:** Create `scripts/cron_retrain.py`

The script:
1. Loads ISS orbit data from `data/raw/`
2. Preprocesses with `OrbitPreprocessor` (stats on `self.stats`)
3. Creates windows with `subsample=2` and reduced `epochs=30` for CPU feasibility
4. Detects architecture from existing checkpoint (hidden_dim, num_layers)
5. Trains `OrbitLSTMDirect` with AdamW + cosine schedule + early stopping
6. Saves as `lstm_iss_6h_retrained.pt` (temporary)
7. Compares val_loss with current best
8. If improved: runs sanity check (predicted ISS distance 6350-6850 km from Earth center)
9. If sanity passes: promotes to `lstm_iss_6h_best.pt` and pushes to HF
10. If anything fails: keeps existing checkpoint, logs warning

- [ ] **Step 5: Write `scripts/cron_retrain.py`** — full script implementing the flow above

- [ ] **Step 6: Test retrain** (30-60 min on CPU)

```bash
cd /var/www/orbit && python3 scripts/cron_retrain.py
```

Expected: Trains 30 epochs, compares with existing, only promotes if improved + sane.

- [ ] **Step 7: Commit**

```bash
git add scripts/cron_retrain.py && git commit -m "feat: add weekly LSTM retrain with sanity checks"
```

---

## Chunk 3: Health Monitoring

### Task 3: Health Check Script

**Files:** Create `scripts/health_check.py`

Six checks:
1. **Site**: `https://orbitalchaos.online` returns 200 (urllib, 10s timeout)
2. **ISS API**: `wheretheiss.at` returns latitude field
3. **NOAA API**: SWPC Kp index endpoint returns data
4. **Predictions**: `predictions.json` exists, < 2 hours old, has 5+ points
5. **Disk**: > 1 GB free (os.statvfs)
6. **Pipeline**: Most recent fetch log < 36 hours old

Uses only stdlib (`urllib.request`, `os`, `json`) — no extra dependencies.
Logs PASS/FAIL for each check. Returns exit code 1 if any fail.
Future: add email/Slack notification on failure.

- [ ] **Step 8: Write `scripts/health_check.py`** — full script implementing all 6 checks

- [ ] **Step 9: Run health check**

```bash
cd /var/www/orbit && python3 scripts/health_check.py
```

Expected: Most checks pass. Predictions may fail if cron hasn't run — that's OK.

- [ ] **Step 10: Commit**

```bash
git add scripts/health_check.py && git commit -m "feat: add health monitoring script"
```

---

## Chunk 4: Cron Registration

### Task 4: Register All Cron Jobs

**Files:** Modify `scripts/cron_setup.sh`

- [ ] **Step 11: Add new cron entries to `scripts/cron_setup.sh`**

Append these lines:
```bash
# Hourly ISS predictions
0 * * * * cd /var/www/orbit && /usr/bin/python3 scripts/cron_predict.py >> logs/predict.log 2>&1

# Weekly LSTM retrain (Sunday 3am, after daily fetch at 2am)
0 3 * * 0 cd /var/www/orbit && /usr/bin/python3 scripts/cron_retrain.py >> logs/retrain.log 2>&1

# Health monitoring every 15 minutes
*/15 * * * * cd /var/www/orbit && /usr/bin/python3 scripts/health_check.py >> logs/health.log 2>&1
```

- [ ] **Step 12: Install cron jobs**

```bash
bash scripts/cron_setup.sh
```

- [ ] **Step 13: Verify cron registered**

```bash
crontab -l | grep orbit
```

Expected: 5 entries (2 existing + 3 new).

- [ ] **Step 14: Wait for next hour, verify prediction ran**

```bash
tail -20 logs/predict.log
```

- [ ] **Step 15: Commit and push**

```bash
git add scripts/cron_setup.sh && git commit -m "feat: register prediction, retrain, and health crons"
git push origin main
```

---

## Session 3 Completion Checklist

- [ ] `scripts/cron_predict.py` — hourly predictions to `public/predictions.json`
- [ ] `scripts/cron_retrain.py` — weekly retrain with sanity checks
- [ ] `scripts/health_check.py` — 6-point health monitoring
- [ ] `scripts/cron_setup.sh` — 3 new cron entries
- [ ] PyTorch CPU + astropy installed
- [ ] All crons registered and verified
- [ ] `predictions.json` generating real data
- [ ] Health checks passing

**Session 4 can now begin.**
