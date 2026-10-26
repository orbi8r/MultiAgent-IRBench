#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
run_training.py — Master script to install deps, test, and kick off full training.

Usage:
    python3 run_training.py [--quick] [--fold on_call] [--epochs 80]
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path


def check_and_install():
    """Install missing packages."""
    required = {
        "torch": "torch --index-url https://download.pytorch.org/whl/cu128",
        "numpy": "numpy",
        "sentence_transformers": "sentence-transformers",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing missing packages: {missing}")
        cmd = [sys.executable, "-m", "pip", "install", "--break-system-packages"] + missing
        subprocess.run(cmd, check=True)


def quick_test():
    """Sanity-check the pipeline with a quick test on one run."""
    print("\n" + "="*60)
    print("Quick sanity test on on_call run...")
    print("="*60)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from method.graph_builder import load_run, build_graph
    from method.features import extract_all_block_features, normalise_features
    from pathlib import Path as P

    run_dir = P("runs/on_call")
    events, blocks, questions = load_run(run_dir)
    g = build_graph(events, blocks)
    total_edges = sum(len(v) for v in g["edges"].values())
    print(f"✓ Graph: {total_edges} edges across 5 types")

    feats = extract_all_block_features(blocks, events, g)
    feats_normed = normalise_features(feats)
    print(f"✓ Features: {len(feats)} blocks × 15 dims")

    import numpy as np
    import torch
    from method.reranker import RLReranker, plackett_luce_sample
    model = RLReranker()
    print(f"✓ RL Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test Plackett-Luce
    scores = torch.randn(10)
    perm, log_prob = plackett_luce_sample(scores, temperature=1.0)
    print(f"✓ Plackett-Luce: perm={perm.tolist()}, log_prob={log_prob.item():.3f}")

    from method.gnn_reranker import GNNReranker
    gnn = GNNReranker()
    print(f"✓ GNN Model: {sum(p.numel() for p in gnn.parameters()):,} parameters")

    print("\nAll checks passed!")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test only (5 epochs, 1 fold)")
    parser.add_argument("--fold", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--gnn_epochs", type=int, default=40)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--no_gnn", action="store_true")
    args = parser.parse_args()

    # Step 1: Check deps
    check_and_install()

    # Step 2: Quick sanity test
    try:
        quick_test()
    except Exception as e:
        print(f"Quick test failed: {e}")
        sys.exit(1)

    # Step 3: Train
    train_args = [sys.executable, "-m", "method.train"]
    if args.quick:
        train_args.append("--quick")
    if args.fold:
        train_args += ["--fold", args.fold]
    train_args += ["--epochs", str(args.epochs)]
    train_args += ["--gnn_epochs", str(args.gnn_epochs)]
    train_args += ["--samples", str(args.samples)]
    if args.no_gnn:
        train_args.append("--no_gnn")

    print(f"\nStarting training: {' '.join(train_args)}")
    result = subprocess.run(train_args)

    if result.returncode != 0:
        print("Training failed!")
        sys.exit(result.returncode)

    # Step 4: Generate plots and evaluation
    print("\nGenerating evaluation plots...")
    eval_args = [sys.executable, "-m", "method.evaluate", "--strata"]
    subprocess.run(eval_args)
    print("\nDone! Check method/results/ for plots and scores.")


if __name__ == "__main__":
    main()
