#!/usr/bin/env python3
"""
Main entry point for GNN Shapley value computation.

Usage:
    python -m gnn.run --methods global_mc lsmr_a --max-samples 500
    python -m gnn.run --methods local_mc lsmr_a --n-jobs 4 --use-reuse
    python -m gnn.run --seeds 42 123 456 --methods global_mc local_mc
"""

import argparse
import os
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from multiprocessing import set_start_method

from .config import (
    CONVERGENCE_THRESHOLD, CHECK_INTERVAL, MAX_SAMPLES,
    CHECKPOINT_INTERVAL, K_HOP, EPOCHS, PRINT_INTERVAL,
)
from .utils import set_seed
from .cache import LocalRegionCache
from .workers import (
    _worker_global_mc, _worker_tmc, _worker_local_mc,
)
from .runners import run_method_standard, run_complementary, run_lsmr_a


# ================================================================
# Method Registry
# ================================================================
METHOD_REGISTRY = {
    "global_mc":     ("Global-MC",   "standard",        _worker_global_mc),
    "tmc":           ("TMC-S",       "standard",        _worker_tmc),
    "complementary": ("Comple-S",    "complementary",   None),
    "local_mc":      ("Local-MC",    "standard",        _worker_local_mc),
    "lsmr_a":        ("LSMR-A",      "lsmr_a",          None),
}
ALL_METHODS = list(METHOD_REGISTRY.keys())


def main():
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Run GNN Shapley value computation to convergence")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Seed for dataset split (default: same as --seeds)")
    parser.add_argument("--method-seed", type=int, default=None,
                        help="Seed for method permutations/model training (default: same as --seeds)")
    parser.add_argument("--k-hop", type=int, default=K_HOP)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--threshold", type=float, default=CONVERGENCE_THRESHOLD)
    parser.add_argument("--check-interval", type=int, default=CHECK_INTERVAL)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--print-interval", type=int, default=PRINT_INTERVAL)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS)
    parser.add_argument("--out-dir", default="gnn_results")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Number of parallel workers (default: auto)")
    parser.add_argument("--train-size", type=int, default=None,
                        help="Number of training nodes (default: all non-test)")
    parser.add_argument("--test-size", type=int, default=1000,
                        help="Number of test nodes")
    parser.add_argument("--data-dir", default="./data",
                        help="Directory for dataset cache (default: ./data)")
    parser.add_argument("--use-reuse", action="store_true", default=False,
                        help="Enable N_z reuse/skip for LSMR-A (default: False)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.n_jobs is not None:
        n_jobs = args.n_jobs
    elif torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        n_jobs = n_gpus * 8
        print(f"Available GPUs: {n_gpus}")
    else:
        n_jobs = 8
    print(f"Using {n_jobs} parallel workers")
    print(f"Seeds: {args.seeds}")

    for seed in args.seeds:
        split_seed = args.split_seed if args.split_seed is not None else seed
        method_seed = args.method_seed if args.method_seed is not None else seed

        print(f"\n{'#'*70}")
        print(f"  SEED = {seed} (split={split_seed}, method={method_seed})")
        print(f"{'#'*70}")

        print(f"\nLoading Cora dataset...")
        set_seed(split_seed)
        dataset = Planetoid(root=args.data_dir, name='Cora')
        data = dataset[0]

        idx = torch.randperm(data.num_nodes)
        test_idx = idx[:args.test_size]
        if args.train_size is not None:
            train_idx = idx[args.test_size:args.test_size + args.train_size]
        else:
            train_idx = idx[args.test_size:]

        n_classes = int(data.y.unique().numel())
        train_list = train_idx.cpu().numpy()
        test_indices = test_idx.cpu().numpy()

        print(f"[Data] Cora: {len(train_list)} train, {len(test_indices)} test, "
              f"{n_classes} classes")

        data_np = {
            'x': data.x.cpu().numpy(),
            'y': data.y.cpu().numpy(),
            'edge_index': data.edge_index.cpu().numpy(),
            'test_indices': test_indices,
        }

        build_N_z = args.use_reuse and "lsmr_a" in args.methods
        cache = LocalRegionCache(
            data.edge_index, train_list, test_indices,
            k_hop=args.k_hop, build_N_z=build_N_z)

        config = {
            'n_classes': n_classes,
            'epochs': args.epochs,
            'conv_threshold': args.threshold,
            'check_interval': args.check_interval,
            'print_interval': args.print_interval,
            'max_samples': args.max_samples,
            'checkpoint_interval': args.checkpoint_interval,
            'seed': method_seed,
            'dataset': "Cora",
            'n_jobs': n_jobs,
            'use_reuse': args.use_reuse,
        }

        os.makedirs(args.out_dir, exist_ok=True)
        results = {}

        for method_key in args.methods:
            display_name, runner_type, worker_fn = METHOD_REGISTRY[method_key]

            print(f"\n{'#'*60}")
            print(f"  {display_name} ({method_key})  [seed={seed}]")
            print(f"{'#'*60}")

            method_out_dir = os.path.join(args.out_dir, method_key)

            if runner_type == "complementary":
                phi, n_samples, conv_history = run_complementary(
                    cache, data_np, config, method_out_dir)
            elif runner_type == "lsmr_a":
                phi, n_samples, conv_history = run_lsmr_a(
                    cache, data_np, config, method_out_dir)
            else:
                phi, n_samples, conv_history = run_method_standard(
                    method_key, worker_fn, cache, data_np, config, method_out_dir)

            results[method_key] = {
                "display_name": display_name,
                "phi": phi,
                "samples": n_samples,
                "convergence_history": conv_history,
            }

        # Summary for this seed
        print(f"\n{'='*70}")
        print(f"SUMMARY  (seed={seed})")
        print(f"{'='*70}")
        print(f"{'Method':<12} {'Samples':>8} {'mean(phi)':>12} {'std(phi)':>12}")
        print("-" * 50)
        for method_key in args.methods:
            r = results[method_key]
            print(f"{r['display_name']:<12} {r['samples']:>8} "
                  f"{r['phi'].mean():>12.6f} {r['phi'].std():>12.6f}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
