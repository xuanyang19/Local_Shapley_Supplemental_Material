import os
import json
import numpy as np


class ConvergenceTracker:
    """Track Shapley value convergence every check_interval samples."""

    def __init__(self, n_points, check_interval=50, threshold=0.05):
        self.n_points = n_points
        self.check_interval = check_interval
        self.threshold = threshold
        self.phi_history = []
        self.convergence_history = []
        self.converged = False

    def check_convergence(self, phi_current, sample_num):
        self.phi_history.append(phi_current.copy())
        result = {"sample": sample_num, "converged": False, "convergence_metric": float('nan')}
        if len(self.phi_history) < 2:
            self.convergence_history.append(result)
            return result
        phi_ago = self.phi_history[-2]
        eps = 1e-12
        relative_change = np.abs(phi_current - phi_ago) / (np.abs(phi_current) + eps)
        avg_convergence = float(np.mean(relative_change))
        result["convergence_metric"] = avg_convergence
        if avg_convergence < self.threshold:
            result["converged"] = True
            self.converged = True
        self.convergence_history.append(result)
        return result


def save_results(phi_hat, num_samples, convergence_history, *, out_dir, tag, dataset, seed):
    """Save phi .npy and JSON stats."""
    os.makedirs(out_dir, exist_ok=True)
    phi_fname = f"{tag}_phi_{dataset}_seed{seed}_n{num_samples}.npy"
    np.save(os.path.join(out_dir, phi_fname), phi_hat)
    stats_fname = f"{tag}_stats_{dataset}_seed{seed}_n{num_samples}.json"
    out = {
        "meta": {"tag": tag, "dataset": dataset, "seed": seed, "num_samples": num_samples},
        "convergence_history": convergence_history,
    }
    with open(os.path.join(out_dir, stats_fname), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  [Saved] {phi_fname}", flush=True)


def maybe_print_progress(tag, s, conv_metric=None, print_interval=50):
    if (s + 1) % print_interval == 0:
        conv_str = (f"conv={conv_metric:.4f}"
                    if conv_metric is not None and not np.isnan(conv_metric)
                    else "conv=N/A")
        print(f"[{tag}] sample={s+1}  {conv_str}", flush=True)


