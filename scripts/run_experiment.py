#!/usr/bin/env python3
"""CLI for running orbit prediction experiments.

Usage:
    python scripts/run_experiment.py --model lstm --spacecraft iss --horizon 6
    python scripts/run_experiment.py --model transformer --spacecraft mms1 --horizon 24
    python scripts/run_experiment.py --model multimodal --spacecraft iss --horizon 6
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.data.ssc_client import SSCClient
from src.data.solar_wind import SolarWindClient
from src.data.preprocessing import OrbitPreprocessor, SolarWindPreprocessor
from src.data.dataset import create_dataloaders
from src.models.lstm import OrbitLSTM, OrbitLSTMDirect
from src.models.transformer import OrbitTransformer, OrbitTransformerDirect
from src.models.baseline_sgp4 import SGP4Baseline, evaluate_baseline
from src.training.train import Trainer
from src.training.evaluate import evaluate_pytorch_model, comparison_table


MODEL_REGISTRY = {
    "lstm": OrbitLSTM,
    "lstm_direct": OrbitLSTMDirect,
    "transformer": OrbitTransformer,
    "transformer_direct": OrbitTransformerDirect,
}


def main():
    parser = argparse.ArgumentParser(description="Run orbit prediction experiment")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()) + ["all", "baseline"],
                        help="Model to train")
    parser.add_argument("--spacecraft", type=str, default="iss",
                        help="Spacecraft ID")
    parser.add_argument("--horizon", type=int, default=6,
                        help="Prediction horizon in hours")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config file path")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # Write updated config back for submodules
    with open(args.config, "w") as f:
        yaml.safe_dump(config, f)

    # Step 1: Load data
    print("Loading data...")
    ssc = SSCClient(args.config)
    sc_config = config["spacecraft"][args.spacecraft]
    orbit_df = ssc.fetch_positions(sc_config["id"], sc_config["start_date"], sc_config["end_date"])

    if len(orbit_df) == 0:
        print("No data available. Run scripts/fetch_data.py first.")
        sys.exit(1)

    # Step 2: Preprocess
    print("Preprocessing...")
    preprocessor = OrbitPreprocessor(args.config)
    orbit_df = preprocessor.preprocess(orbit_df, args.spacecraft)

    time_res = config["model"]["time_resolution_minutes"]
    horizon_steps = (args.horizon * 60) // time_res

    inputs, targets, timestamps = preprocessor.create_sliding_windows(
        orbit_df, horizon_hours=args.horizon
    )
    print(f"  Windows: {inputs.shape[0]}, Input: {inputs.shape[1]} steps, Target: {targets.shape[1]} steps")

    # Step 3: Split
    splits = preprocessor.temporal_split(inputs, targets, timestamps)
    for name, (x, y) in splits.items():
        print(f"  {name}: {x.shape[0]} samples")

    input_dim = inputs.shape[-1]
    output_dim = targets.shape[-1]

    # Step 4: Train
    results = {}

    if args.model == "baseline" or args.model == "all":
        print("\nEvaluating Kepler baseline...")
        test_inputs, test_targets = splits["test"]
        # Use last position and velocity from input as initial state
        test_pos = test_inputs[:, -1, :3]
        test_vel = test_inputs[:, -1, 3:6] if input_dim >= 6 else np.zeros_like(test_pos)
        baseline_metrics = evaluate_baseline(test_pos, test_vel, test_targets, dt_seconds=60)
        results["Kepler"] = baseline_metrics
        print(f"  Kepler MAE: {baseline_metrics['mae_km']:.2f} km")

    models_to_train = (
        list(MODEL_REGISTRY.keys()) if args.model == "all"
        else [args.model] if args.model != "baseline"
        else []
    )

    for model_name in models_to_train:
        print(f"\nTraining {model_name}...")
        model_cls = MODEL_REGISTRY[model_name]
        model = model_cls(
            input_dim=input_dim,
            horizon=horizon_steps,
            output_dim=output_dim,
        )

        loaders = create_dataloaders(splits, batch_size=config["training"]["batch_size"])
        trainer = Trainer(model, args.config)
        trainer.train(loaders["train"], loaders["val"], model_name=f"{model_name}_{args.spacecraft}")

        denorm_fn = lambda x: preprocessor.denormalize(x, args.spacecraft)
        metrics = evaluate_pytorch_model(model, loaders["test"], denormalize_fn=denorm_fn)
        results[model_name] = metrics

    # Step 5: Compare
    if results:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(comparison_table(results))


if __name__ == "__main__":
    main()
