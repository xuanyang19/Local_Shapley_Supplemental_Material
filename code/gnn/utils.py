#!/usr/bin/env python3
"""
Shared utilities for GNN Shapley value computation.
- ConvergenceTracker
- I/O (save_results, save_checkpoint)
- set_seed, progress printing
"""

import numpy as np
import torch
import os
import json
import datetime as _dt


# ================================================================
# Convergence Tracker
# ================================================================
class ConvergenceTracker:
    def __init__(self, n_points, check_interval=100, threshold=0.05):
        self.n_points = n_points
        self.check_interval = check_interval
        self.threshold = threshold
        self.phi_history = []
        self.convergence_history = []
        self.converged = False

    def check_convergence(self, phi_current, sample_num):
        self.phi_history.append(phi_current.copy())
        result = {"sample": sample_num, "converged": False,
                  "convergence_metric": float('nan')}

        if len(self.phi_history) < 2:
            self.convergence_history.append(result)
            return result

        phi_now = phi_current
        phi_ago = self.phi_history[-2]
        eps = 1e-12
        relative_change = np.abs(phi_now - phi_ago) / (np.abs(phi_now) + eps)
        avg_convergence = float(np.mean(relative_change))
        result["convergence_metric"] = avg_convergence

        if avg_convergence < self.threshold:
            result["converged"] = True
            self.converged = True

        self.convergence_history.append(result)
        return result


# ================================================================
# I/O Helpers
# ================================================================
def _timestamp():
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_checkpoint(phi_sum, num_samples, convergence_history,
                    *, out_dir, tag, dataset, seed, n_test):
    os.makedirs(out_dir, exist_ok=True)
    phi_hat = phi_sum / float(num_samples)
    phi_hat_accuracy = phi_hat / float(n_test)

    phi_fname = f"{tag}_phi_{dataset}_seed{seed}_n{num_samples}_ckpt.npy"
    np.save(os.path.join(out_dir, phi_fname), phi_hat_accuracy)

    stats_fname = f"{tag}_stats_{dataset}_seed{seed}_n{num_samples}_ckpt.json"
    out = {
        "meta": {
            "tag": tag, "dataset": dataset, "seed": seed,
            "num_samples": num_samples, "created_at": _timestamp(),
            "n_test": n_test, "scale": "accuracy",
            "is_checkpoint": True,
        },
        "convergence_history": convergence_history,
    }
    with open(os.path.join(out_dir, stats_fname), "w") as f:
        json.dump(out, f, indent=2)

    print(f"  [Checkpoint] Saved at sample {num_samples}: {phi_fname}", flush=True)


def save_results(phi_hat, num_samples, convergence_history,
                 *, out_dir, tag, dataset, seed, n_test):
    os.makedirs(out_dir, exist_ok=True)
    phi_hat_accuracy = phi_hat / float(n_test)

    phi_fname = f"{tag}_phi_{dataset}_seed{seed}_n{num_samples}.npy"
    np.save(os.path.join(out_dir, phi_fname), phi_hat_accuracy)

    stats_fname = f"{tag}_stats_{dataset}_seed{seed}_n{num_samples}.json"
    out = {
        "meta": {
            "tag": tag, "dataset": dataset, "seed": seed,
            "num_samples": num_samples, "created_at": _timestamp(),
            "n_test": n_test, "scale": "accuracy",
        },
        "convergence_history": convergence_history,
    }
    with open(os.path.join(out_dir, stats_fname), "w") as f:
        json.dump(out, f, indent=2)

    print(f"  [Saved] {phi_fname}, {stats_fname} (accuracy scale, n_test={n_test})", flush=True)
    return num_samples


# ================================================================
# Seed & Progress
# ================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_print_progress(tag, s, conv_metric=None, print_interval=10):
    if (s + 1) % print_interval == 0:
        conv_str = (f"conv={conv_metric:.4f}"
                    if conv_metric is not None and not np.isnan(conv_metric)
                    else "conv=N/A")
        print(f"[{tag}] sample={s+1}  {conv_str}", flush=True)
