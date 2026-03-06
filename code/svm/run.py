import argparse
import numpy as np
from .config import CONVERGENCE_THRESHOLD, CHECK_INTERVAL, MAX_SAMPLES, N_JOBS, KERNEL_THRESHOLD
from .helpers import compute_gamma
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
    from data import load_data, set_seed

    parser = argparse.ArgumentParser(description="SVM Shapley - Run to Convergence")
    parser.add_argument("--dataset", default="Breast_Cancer")
    parser.add_argument("--sample-size", type=int, nargs=2, default=[1000, 1000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Seed for dataset split/sampling (default: same as --seeds)")
    parser.add_argument("--method-seed", type=int, default=None,
                        help="Seed for method permutations/model training (default: same as --seeds)")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--kernel-threshold", type=float, default=KERNEL_THRESHOLD)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--threshold", type=float, default=CONVERGENCE_THRESHOLD)
    parser.add_argument("--check-interval", type=int, default=CHECK_INTERVAL)
    parser.add_argument("--print-interval", type=int, default=50)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--out-dir", default="svm_results")
    parser.add_argument("--no-reuse", dest="use_reuse", action="store_false", default=True,
                        help="Disable T_S reuse/skip in LSMR-A (per-test-point mode)")
    args = parser.parse_args()

    for seed in args.seeds:
        split_seed = args.split_seed if args.split_seed is not None else seed
        method_seed = args.method_seed if args.method_seed is not None else seed

        print(f"\n{'='*70}")
        print(f"SEED: {seed} (split={split_seed}, method={method_seed})")
        print(f"{'='*70}")
        set_seed(split_seed)
        X_train, y_train, X_test, y_test = load_data(
            args.dataset, sample_size=args.sample_size, random_state=split_seed)
        print(f"[Data] {args.dataset}: {len(X_train)} train, {len(X_test)} test")

        gamma = compute_gamma(X_train)
        print(f"[SVM] gamma = {gamma:.6f}, kernel_threshold = {args.kernel_threshold}")

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
                gamma=gamma,
                kernel_threshold=args.kernel_threshold,
            )

            if method_key in ("global_mc", "tmc", "complementary"):
                kwargs["n_jobs"] = args.n_jobs
            if method_key == "tmc":
                kwargs["patience"] = args.patience
            if method_key == "lsmr_a":
                kwargs["use_reuse"] = args.use_reuse

            phi, n_samples, conv_hist = method_fn(
                X_train, y_train, X_test, y_test, **kwargs)
            results[method_key] = {
                "display": display_name, "samples": n_samples, "phi": phi}

        print(f"\n{'='*60}")
        print(f"SUMMARY (seed={seed})")
        print(f"{'='*60}")
        for k in args.methods:
            r = results[k]
            print(f"  {r['display']:<12} samples={r['samples']:>6}  "
                  f"mean={r['phi'].mean():.6f}  std={r['phi'].std():.6f}")

        if "global_mc" in results and len(results) > 1:
            try:
                from scipy.stats import pearsonr
                phi_ref = results["global_mc"]["phi"]
                print(f"\nPearson r vs Global-MC:")
                for k in args.methods:
                    if k == "global_mc":
                        continue
                    r_val, _ = pearsonr(phi_ref, results[k]["phi"])
                    print(f"  {results[k]['display']:<12} r = {r_val:.4f}")
            except ImportError:
                pass

    print("\n[Done]")


if __name__ == "__main__":
    main()
