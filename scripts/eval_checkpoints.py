#!/usr/bin/env python3
"""Re-evaluate saved checkpoints with fixed denormalization."""

import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import pandas as pd
import yaml

from src.data.preprocessing import OrbitPreprocessor
from src.data.dataset import OrbitDataset
from src.models.lstm import OrbitLSTMDirect
from src.models.transformer import OrbitTransformerDirect


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    sc = "iss"
    sc_config = config["spacecraft"][sc]
    subsample = 10

    # Reload + preprocess
    df = pd.read_parquet(f"data/raw/{sc}_{sc_config['start_date']}_{sc_config['end_date']}.parquet")
    prep = OrbitPreprocessor()
    processed = prep.preprocess(df, sc)

    inputs, targets, timestamps = prep.create_sliding_windows(
        processed, input_hours=24, horizon_hours=6, stride_hours=6, subsample=subsample
    )
    splits = prep.temporal_split(inputs, targets, timestamps)
    test_in, test_tgt = splits["test"]

    print(f"Test set: {len(test_in)} windows, input: {test_in.shape}, target: {test_tgt.shape}")
    print(f"Stats keys: {list(prep.stats[sc]['mean'].keys())}")

    test_loader = torch.utils.data.DataLoader(
        OrbitDataset(test_in, test_tgt), batch_size=32
    )

    input_dim = test_in.shape[-1]
    output_dim = test_tgt.shape[-1]
    horizon = test_tgt.shape[1]

    for name, ModelClass, kwargs in [
        ("LSTM", OrbitLSTMDirect, dict(input_dim=input_dim, hidden_dim=64, num_layers=2,
                                        horizon=horizon, output_dim=output_dim, dropout=0.1)),
        ("Transformer", OrbitTransformerDirect, dict(input_dim=input_dim, d_model=64, nhead=4,
                                                      num_layers=2, dim_feedforward=128,
                                                      horizon=horizon, output_dim=output_dim, dropout=0.1)),
    ]:
        ckpt_path = Path(f"checkpoints/{name.lower()}_{sc}_6h_best.pt")
        if not ckpt_path.exists():
            print(f"\n{name}: No checkpoint found")
            continue

        model = ModelClass(**kwargs)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x)
                ml = min(pred.shape[1], y.shape[1])
                mf = min(pred.shape[2], y.shape[2])
                all_preds.append(pred[:, :ml, :mf].numpy())
                all_targets.append(y[:, :ml, :mf].numpy())

        preds = np.concatenate(all_preds)
        tgts = np.concatenate(all_targets)

        # Denormalize
        preds_km = prep.denormalize(preds, sc)
        tgts_km = prep.denormalize(tgts, sc)

        distances = np.sqrt(np.sum((preds_km - tgts_km)**2, axis=-1))
        mae = np.mean(distances)
        rmse = np.sqrt(np.mean(distances**2))

        print(f"\n{'='*60}")
        print(f"{name} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
        print(f"{'='*60}")
        print(f"  Overall MAE:  {mae:.2f} km")
        print(f"  Overall RMSE: {rmse:.2f} km")

        n_steps = distances.shape[1]
        for frac, label in [(0.25, "1.5h"), (0.5, "3h"), (1.0, "6h")]:
            idx = min(int(frac * n_steps) - 1, n_steps - 1)
            h_mae = np.mean(distances[:, idx])
            h_rmse = np.sqrt(np.mean(distances[:, idx]**2))
            print(f"  @{label} (step {idx+1}/{n_steps}): MAE={h_mae:.2f} km, RMSE={h_rmse:.2f} km")


if __name__ == "__main__":
    main()
