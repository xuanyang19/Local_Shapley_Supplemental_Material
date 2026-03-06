"""
Entry point for template Shapley computation.

Usage:
    python -m template --model-module template.examples.svm_model --dataset Iris --methods global_mc lsmr_a
"""

import argparse
import importlib
import numpy as np
from .config import CONVERGENCE_THRESHOLD, CHECK_INTERVAL, MAX_SAMPLES
from data import set_seed
from .methods import (
    shapley_global_mc, shapley_tmc, shapley_complementary,
    shapley_local_mc, shapley_lsmr_a,
)

METHOD_REGISTRY = {
    "global_mc":     ("Global-MC", shapley_global_mc),
    "tmc":           ("TMC-S",     shapley_tmc),
    "complementary": ("Comple-S",  shapley_complementary),
    "local_mc":      ("Local-MC",  shapley_local_mc),
    "lsmr_a":        ("LSMR-A",    shapley_lsmr_a),
}
ALL_METHODS = list(METHOD_REGISTRY.keys())


def main():
    parser = argparse.ArgumentParser(description="Template Shapley - Run to Convergence")
    parser.add_argument("--model-module", required=True,
                        help="Python module path providing setup_model(args) -> (model, local_region, n_train, n_test)")
    parser.add_argument("--dataset", default="Iris")
    parser.add_argument("--sample-size", type=int, nargs=2, default=[100, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--method-seed", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--threshold", type=float, default=CONVERGENCE_THRESHOLD)
    parser.add_argument("--check-interval", type=int, default=CHECK_INTERVAL)
    parser.add_argument("--print-interval", type=int, default=50)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--out-dir", default="template_results")
    parser.add_argument("--use-reuse", type=lambda v: v.lower() in ("true", "1", "yes"),
                        default=True, metavar="BOOL",
                        help="LSMR-A: enable T_S reuse/pivot (default: True)")
    args = parser.parse_args()

    # Import user-provided model module
    mod = importlib.import_module(args.model_module)

    for seed in args.seeds:
        split_seed = args.split_seed if args.split_seed is not None else seed
        method_seed = args.method_seed if args.method_seed is not None else seed

        print(f"\n{'='*70}")
        print(f"SEED: {seed} (split={split_seed}, method={method_seed})")
        print(f"{'='*70}")
        set_seed(split_seed)

        # User module must provide setup_model(args) -> (model, local_region, n_train, n_test)
        model, local_region, n_train, n_test = mod.setup_model(args, split_seed)

        results = {}
        for method_key in args.methods:
            display_name, method_fn = METHOD_REGISTRY[method_key]

            kwargs = dict(
                max_samples=args.max_samples, seed=method_seed,
                print_interval=args.print_interval,
                out_dir=f"{args.out_dir}/{method_key}",
                dataset=args.dataset,
                convergence_threshold=args.threshold,
                check_interval=args.check_interval,
            )

            if method_key in ("global_mc", "tmc", "complementary"):
                kwargs["n_train"] = n_train
                kwargs["n_test"] = n_test
            if method_key == "tmc":
                kwargs["patience"] = args.patience

            if method_key == "lsmr_a":
                kwargs["use_reuse"] = args.use_reuse
            if method_key in ("local_mc", "lsmr_a"):
                phi, n_samples, conv_hist = method_fn(model, local_region, **kwargs)
            elif method_key in ("global_mc", "tmc", "complementary"):
                phi, n_samples, conv_hist = method_fn(model, **kwargs)
            else:
                phi, n_samples, conv_hist = method_fn(model, **kwargs)

            results[method_key] = {"display": display_name, "samples": n_samples, "phi": phi}

        print(f"\n{'='*60}")
        print(f"SUMMARY (seed={seed})")
        print(f"{'='*60}")
        for k in args.methods:
            r = results[k]
            print(f"  {r['display']:<12} samples={r['samples']:>6}  "
                  f"mean={r['phi'].mean():.6f}  std={r['phi'].std():.6f}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
