#!/usr/bin/env python3
"""Five WKNN Shapley value methods: Global MC, TMC, Complementary, Local MC, LSMR-A."""

import numpy as np

from .config import CONVERGENCE_THRESHOLD, CHECK_INTERVAL, MAX_SAMPLES, K_NEIGHBORS
from .utils import ConvergenceTracker, save_results, maybe_print_progress
from .helpers import (
    _wknn_predict_manual,
    _compute_utility_naive,
    _compute_utility_local_naive,
    precompute_all_distances,
    precompute_knn_neighbors,
    precompute_knn_neighbors_with_T_z,
    build_log_factorial_table,
    importance_weight,
)


# ============================================================
# 1. GLOBAL MC
# ============================================================
def shapley_global_mc(
    X_tr, y_tr, X_te, y_te, *,
    max_samples=MAX_SAMPLES, seed=42, print_interval=100,
    out_dir="wknn_results", dataset="UNK", k=K_NEIGHBORS,
    convergence_threshold=CONVERGENCE_THRESHOLD, check_interval=CHECK_INTERVAL
):
    """Global MC Shapley for WKNN with convergence-based stopping."""
    tag = "global_mc"
    n_tr, n_te = len(X_tr), len(X_te)
    n_classes = len(np.unique(y_tr))

    conv_tracker = ConvergenceTracker(n_tr, check_interval, convergence_threshold)
    phi_sum = np.zeros(n_tr, dtype=float)
    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} WKNN (K={k})...")
    print(f"Convergence: threshold={convergence_threshold}, check_interval={check_interval}")
    print(f"Precomputing distances...")
    print(f"{'='*60}")

    dist_matrix = precompute_all_distances(X_tr, X_te)

    num_samples_done = 0

    for m in range(max_samples):
        contrib = np.zeros(n_tr, dtype=float)

        for t in range(n_te):
            y_t = y_te[t]
            dist_vec = dist_matrix[t]
            pi = rng.permutation(n_tr).astype(int)
            v_prev = 1.0 / n_classes

            coalition = []
            for z_int in pi.tolist():
                coalition.append(z_int)

                if len(coalition) <= k:
                    topk = coalition[:]
                else:
                    topk = sorted(coalition, key=lambda idx: dist_vec[idx])[:k]

                pred = _wknn_predict_manual(topk, dist_vec, y_tr, n_classes)
                v_curr = 1.0 if pred == y_t else 0.0

                contrib[z_int] += (v_curr - v_prev)
                v_prev = v_curr

        phi_sum += contrib
        num_samples_done = m + 1

        conv_metric = None
        if num_samples_done % check_interval == 0:
            phi_current = phi_sum / float(num_samples_done)
            conv_result = conv_tracker.check_convergence(phi_current, num_samples_done)
            conv_metric = conv_result["convergence_metric"]

            if conv_result["converged"]:
                print(f"\n[{tag}] CONVERGED at sample {num_samples_done}! "
                      f"metric={conv_metric:.6f} < {convergence_threshold}")
                break

        maybe_print_progress(tag, m, conv_metric, print_interval)

    phi_hat = phi_sum / float(num_samples_done)
    save_results(phi_hat, num_samples_done, conv_tracker.convergence_history,
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed)

    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ============================================================
# 2. TMC-Shapley
# ============================================================
def shapley_tmc(
    X_tr, y_tr, X_te, y_te, *,
    max_samples=MAX_SAMPLES, seed=42, patience=5, print_interval=100,
    out_dir="wknn_results", dataset="UNK", k=K_NEIGHBORS,
    convergence_threshold=CONVERGENCE_THRESHOLD, check_interval=CHECK_INTERVAL
):
    """TMC-Shapley for WKNN with convergence-based stopping."""
    tag = "tmc"
    n_tr, n_te = len(X_tr), len(X_te)
    n_classes = len(np.unique(y_tr))

    conv_tracker = ConvergenceTracker(n_tr, check_interval, convergence_threshold)
    phi_sum = np.zeros(n_tr, dtype=float)

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} WKNN (K={k}, patience={patience})...")
    print(f"Convergence: threshold={convergence_threshold}, check_interval={check_interval}")
    print(f"Precomputing distances...")
    print(f"{'='*60}")

    dist_matrix = precompute_all_distances(X_tr, X_te)

    # Precompute full model predictions for early stopping
    yhat_full = np.zeros(n_te, dtype=int)
    for t in range(n_te):
        dist_vec = dist_matrix[t]
        topk = list(np.argsort(dist_vec)[:k])
        yhat_full[t] = _wknn_predict_manual(topk, dist_vec, y_tr, n_classes)

    rng = np.random.default_rng(seed)

    num_samples_done = 0

    for m in range(max_samples):
        contrib = np.zeros(n_tr, dtype=float)

        for t in range(n_te):
            y_t = y_te[t]
            full_pred = yhat_full[t]
            dist_vec = dist_matrix[t]
            pi = rng.permutation(n_tr).astype(int)
            v_prev = 1.0 / n_classes
            stable_count = 0
            last_pred = None

            coalition = []
            for z_int in pi.tolist():
                coalition.append(z_int)

                if len(coalition) <= k:
                    topk = coalition[:]
                else:
                    topk = sorted(coalition, key=lambda idx: dist_vec[idx])[:k]

                pred = _wknn_predict_manual(topk, dist_vec, y_tr, n_classes)
                v_curr = 1.0 if pred == y_t else 0.0

                contrib[z_int] += (v_curr - v_prev)
                v_prev = v_curr

                # TMC early stopping
                if pred == full_pred:
                    stable_count = (stable_count + 1) if last_pred == pred else 1
                    if stable_count >= patience and len(coalition) >= k:
                        break
                else:
                    stable_count = 0
                last_pred = pred

        phi_sum += contrib
        num_samples_done = m + 1

        conv_metric = None
        if num_samples_done % check_interval == 0:
            phi_current = phi_sum / float(num_samples_done)
            conv_result = conv_tracker.check_convergence(phi_current, num_samples_done)
            conv_metric = conv_result["convergence_metric"]

            if conv_result["converged"]:
                print(f"\n[{tag}] CONVERGED at sample {num_samples_done}! "
                      f"metric={conv_metric:.6f} < {convergence_threshold}")
                break

        maybe_print_progress(tag, m, conv_metric, print_interval)

    phi_hat = phi_sum / float(num_samples_done)
    save_results(phi_hat, num_samples_done, conv_tracker.convergence_history,
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed)

    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ============================================================
# 3. Complementary Global (Comple-S) - EXACT Algorithm 2
# ============================================================
def shapley_complementary(
    X_tr, y_tr, X_te, y_te, *,
    max_samples=MAX_SAMPLES, seed=42, print_interval=100,
    out_dir="wknn_results", dataset="UNK", k=K_NEIGHBORS,
    convergence_threshold=CONVERGENCE_THRESHOLD, check_interval=CHECK_INTERVAL
):
    """
    Complementary Shapley - EXACT Algorithm 2.
    i = (m % n) + 1 (deterministic cycling).
    2D bins: (n_tr, n_tr).
    """
    tag = "complementary"
    n_tr, n_te = len(X_tr), len(X_te)
    n_classes = len(np.unique(y_tr))

    conv_tracker = ConvergenceTracker(n_tr, check_interval, convergence_threshold)

    SV_bins = np.zeros((n_tr, n_tr), dtype=float)
    M_bins = np.zeros((n_tr, n_tr), dtype=int)

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} WKNN (K={k}) - EXACT Algorithm 2...")
    print(f"Using deterministic i = (m % n) + 1")
    print(f"Convergence: threshold={convergence_threshold}, check_interval={check_interval}")
    print(f"Precomputing distances...")
    print(f"{'='*60}")

    dist_matrix = precompute_all_distances(X_tr, X_te)
    rng = np.random.default_rng(seed)

    num_samples_done = 0

    for m in range(max_samples):
        # EXACT Algorithm 2: deterministic i cycling
        i = (m % n_tr) + 1  # i in [1, n_tr], S is NEVER empty

        for t in range(n_te):
            y_t = y_te[t]
            dist_vec = dist_matrix[t]

            pi = rng.permutation(n_tr).astype(int)
            S = pi[:i].tolist()
            S_comp = pi[i:].tolist()

            j_plus = i - 1
            j_minus = n_tr - i - 1

            v_S = _compute_utility_naive(S, dist_vec, y_tr, y_t, k, n_classes)

            if len(S_comp) == 0:
                v_Sc = 1.0 / n_classes
            else:
                v_Sc = _compute_utility_naive(S_comp, dist_vec, y_tr, y_t, k, n_classes)

            u_t = v_S - v_Sc

            # Update bins
            SV_bins[S, j_plus] += u_t
            M_bins[S, j_plus] += 1

            if len(S_comp) > 0 and j_minus >= 0:
                SV_bins[S_comp, j_minus] -= u_t
                M_bins[S_comp, j_minus] += 1

        num_samples_done = m + 1

        conv_metric = None
        if num_samples_done % check_interval == 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                means_now = np.where(M_bins > 0, SV_bins / np.maximum(M_bins, 1), 0.0)
            phi_current = means_now.mean(axis=1)
            conv_result = conv_tracker.check_convergence(phi_current, num_samples_done)
            conv_metric = conv_result["convergence_metric"]

            if conv_result["converged"]:
                print(f"\n[{tag}] CONVERGED at sample {num_samples_done}! "
                      f"metric={conv_metric:.6f} < {convergence_threshold}")
                break

        maybe_print_progress(tag, m, conv_metric, print_interval)

    # Final phi: bin-based stratified estimator
    with np.errstate(divide='ignore', invalid='ignore'):
        means_per_bin = np.where(M_bins > 0, SV_bins / np.maximum(M_bins, 1), 0.0)
    phi_cc = means_per_bin.mean(axis=1)

    save_results(phi_cc, num_samples_done, conv_tracker.convergence_history,
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed)

    return phi_cc, num_samples_done, conv_tracker.convergence_history


# ============================================================
# 4. LOCAL MC
# ============================================================
def shapley_local_mc(
    X_train, y_train, X_test, y_test, *,
    max_samples=MAX_SAMPLES, seed=42, print_interval=100,
    out_dir="wknn_results", dataset="UNK",
    k=K_NEIGHBORS, k_local=20,
    convergence_threshold=CONVERGENCE_THRESHOLD, check_interval=CHECK_INTERVAL
):
    """Local MC Shapley for WKNN with convergence-based stopping."""
    tag = "local_mc"
    n_train, n_test = len(X_train), len(X_test)
    n_classes = len(np.unique(y_train))

    conv_tracker = ConvergenceTracker(n_train, check_interval, convergence_threshold)
    phi_sum = np.zeros(n_train, dtype=float)

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} WKNN (K={k}, k_local={k_local})...")
    print(f"Convergence: threshold={convergence_threshold}, check_interval={check_interval}")
    print(f"Precomputing KNN-based local neighborhoods...")
    print(f"{'='*60}")

    neighbors_list, distances_list = precompute_knn_neighbors(X_train, X_test, k_local)

    local_sizes = [len(neighbors_list[t]) for t in range(n_test)]
    print(f"Local region sizes: min={min(local_sizes)}, max={max(local_sizes)}, mean={np.mean(local_sizes):.1f}")

    rng = np.random.default_rng(seed)

    num_samples_done = 0

    for m in range(max_samples):
        contrib = np.zeros(n_train, dtype=float)

        for t in range(n_test):
            L = neighbors_list[t]
            L_len = len(L)
            if L_len == 0:
                continue

            y_t = y_test[t]
            dist_local = distances_list[t]
            y_local = y_train[L]

            perm_pos = rng.permutation(L_len)
            perm = L[perm_pos]

            v_prev = 1.0 / n_classes

            coalition_pos = []
            for idx, z in enumerate(perm):
                local_pos = perm_pos[idx]
                coalition_pos.append(local_pos)

                if len(coalition_pos) <= k:
                    topk_pos = coalition_pos[:]
                else:
                    topk_pos = sorted(coalition_pos, key=lambda p: dist_local[p])[:k]

                pred = _wknn_predict_manual(topk_pos, dist_local, y_local, n_classes)
                v_curr = 1.0 if pred == y_t else 0.0

                contrib[z] += (v_curr - v_prev)
                v_prev = v_curr

        phi_sum += contrib
        num_samples_done = m + 1

        conv_metric = None
        if num_samples_done % check_interval == 0:
            phi_current = phi_sum / float(num_samples_done)
            conv_result = conv_tracker.check_convergence(phi_current, num_samples_done)
            conv_metric = conv_result["convergence_metric"]

            if conv_result["converged"]:
                print(f"\n[{tag}] CONVERGED at sample {num_samples_done}! "
                      f"metric={conv_metric:.6f} < {convergence_threshold}")
                break

        maybe_print_progress(tag, m, conv_metric, print_interval)

    phi_hat = phi_sum / float(num_samples_done)
    save_results(phi_hat, num_samples_done, conv_tracker.convergence_history,
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed)

    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ============================================================
# 5. LSMR-A (Local Complementary with optional T_S reuse)
# ============================================================
def shapley_lsmr_a(
    X_train, y_train, X_test, y_test, *,
    max_samples=MAX_SAMPLES, seed=42, print_interval=100,
    out_dir="wknn_results", dataset="UNK",
    k=K_NEIGHBORS, k_local=20,
    convergence_threshold=CONVERGENCE_THRESHOLD, check_interval=CHECK_INTERVAL,
    use_reuse=False
):
    tag = "lsmr_a"
    n_train, n_test = len(X_train), len(X_test)
    n_classes = len(np.unique(y_train))

    conv_tracker = ConvergenceTracker(n_train, check_interval, convergence_threshold)

    # Per-test-point bins: (n_test, k_local, k_local)
    SV_bins = np.zeros((n_test, k_local, k_local), dtype=float)
    M_bins = np.zeros((n_test, k_local, k_local), dtype=int)
    phi_sum = np.zeros(n_train, dtype=float)  # for convergence tracking

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} WKNN (K={k}, k_local={k_local}, reuse={use_reuse})...")
    print(f"Convergence: threshold={convergence_threshold}, check_interval={check_interval}")
    print(f"Precomputing KNN-based local neighborhoods...")
    print(f"{'='*60}")

    if use_reuse:
        neighbors_list, distances_list, T_z = precompute_knn_neighbors_with_T_z(
            X_train, X_test, k_local)
        L_sizes = np.array([len(neighbors_list[t]) for t in range(n_test)], dtype=int)
        log_fact = build_log_factorial_table(int(L_sizes.max()) + 1)
        Pi_T = np.arange(n_test, dtype=int)
        order = {t: i for i, t in enumerate(Pi_T)}
        T_all = set(range(n_test))
    else:
        neighbors_list, distances_list = precompute_knn_neighbors(X_train, X_test, k_local)
        L_sizes = np.array([len(neighbors_list[t]) for t in range(n_test)], dtype=int)

    print(f"Local region sizes: min={min(L_sizes)}, max={max(L_sizes)}, mean={np.mean(L_sizes):.1f}")

    rng = np.random.default_rng(seed)
    num_samples_done = 0

    for m in range(max_samples):
        contrib_iter = np.zeros(n_train, dtype=float)

        test_iter = Pi_T if use_reuse else range(n_test)
        for t in test_iter:
            L = neighbors_list[t]
            L_len = len(L)
            if L_len == 0:
                continue

            perm_pos = rng.permutation(L_len)
            i = int(rng.integers(0, L_len + 1))

            S_pos = perm_pos[:i].tolist()
            Sc_pos = perm_pos[i:].tolist()

            # --- T_S pivot/skip (when reuse enabled) ---
            if use_reuse:
                S_global = [int(L[p]) for p in S_pos] if i > 0 else []
                if i == 0:
                    T_S = T_all
                else:
                    sets = sorted([T_z[z] for z in S_global], key=len)
                    inter = set(sets[0])
                    for s in sets[1:]:
                        inter &= s
                        if not inter:
                            break
                    T_S = inter

                if not T_S:
                    continue
                t_star = min(T_S, key=lambda x: order[x])
                if t != t_star:
                    continue  # skip: not the pivot

                # Process all test points in T_S
                T_S_arr = sorted(T_S, key=lambda x: order[x])
                for t_prime in T_S_arr:
                    L_tp = neighbors_list[t_prime]
                    L_tp_len = len(L_tp)
                    dist_local_tp = distances_list[t_prime]
                    y_local_tp = y_train[L_tp]
                    y_tp = y_test[t_prime]

                    # Map S_pos/Sc_pos from L_t positions to L_t' positions
                    # S_global indices must be looked up in L_tp
                    tp_idx_map = {int(L_tp[p]): p for p in range(L_tp_len)}
                    S_pos_tp = [tp_idx_map[z] for z in S_global if z in tp_idx_map] if i > 0 else []
                    Sc_global = [int(L[p]) for p in Sc_pos]
                    Sc_pos_tp = [tp_idx_map[z] for z in Sc_global if z in tp_idx_map]

                    v_S = _compute_utility_local_naive(S_pos_tp, dist_local_tp, y_local_tp, y_tp, k, n_classes)
                    v_Sc = _compute_utility_local_naive(Sc_pos_tp, dist_local_tp, y_local_tp, y_tp, k, n_classes)
                    u_tp = v_S - v_Sc

                    w = importance_weight(L_len, L_tp_len, i, log_fact) if L_len != L_tp_len and i > 0 else 1.0
                    u_weighted = u_tp * w

                    j_plus = i - 1
                    j_minus = L_tp_len - i - 1

                    if i > 0 and j_plus < k_local:
                        SV_bins[t_prime, S_pos_tp, j_plus] += u_weighted
                        M_bins[t_prime, S_pos_tp, j_plus] += 1
                    if L_tp_len - i > 0 and j_minus >= 0 and j_minus < k_local:
                        SV_bins[t_prime, Sc_pos_tp, j_minus] -= u_weighted
                        M_bins[t_prime, Sc_pos_tp, j_minus] += 1

                    # Convergence tracking (biased phi_sum)
                    S_global_tp = [int(L_tp[p]) for p in S_pos_tp]
                    Sc_global_tp = [int(L_tp[p]) for p in Sc_pos_tp]
                    if S_global_tp:
                        contrib_iter[S_global_tp] += u_weighted
                    if Sc_global_tp:
                        contrib_iter[Sc_global_tp] -= u_weighted
            else:
                # --- No reuse: original per-test-point computation ---
                dist_local = distances_list[t]
                y_local = y_train[L]
                y_t = y_test[t]

                j_plus = i - 1
                j_minus = L_len - i - 1

                v_S = _compute_utility_local_naive(S_pos, dist_local, y_local, y_t, k, n_classes)
                v_Sc = _compute_utility_local_naive(Sc_pos, dist_local, y_local, y_t, k, n_classes)
                u = v_S - v_Sc

                S_global = [L[p] for p in S_pos] if i > 0 else []
                Sc_global = [L[p] for p in Sc_pos] if L_len - i > 0 else []

                if i > 0:
                    contrib_iter[S_global] += u
                if L_len - i > 0:
                    contrib_iter[Sc_global] -= u

                if i > 0 and j_plus < k_local:
                    SV_bins[t, S_pos, j_plus] += u
                    M_bins[t, S_pos, j_plus] += 1
                if L_len - i > 0 and j_minus >= 0 and j_minus < k_local:
                    SV_bins[t, Sc_pos, j_minus] -= u
                    M_bins[t, Sc_pos, j_minus] += 1

        phi_sum += contrib_iter
        num_samples_done = m + 1

        conv_metric = None
        if num_samples_done % check_interval == 0:
            phi_current = phi_sum / float(num_samples_done)
            conv_result = conv_tracker.check_convergence(phi_current, num_samples_done)
            conv_metric = conv_result["convergence_metric"]

            if conv_result["converged"]:
                print(f"\n[{tag}] CONVERGED at sample {num_samples_done}! "
                      f"metric={conv_metric:.6f} < {convergence_threshold}")
                break

        maybe_print_progress(tag, m, conv_metric, print_interval)

    # Final phi: unbiased per-test-point binned Shapley, summed across test points
    with np.errstate(divide='ignore', invalid='ignore'):
        bin_means = np.where(M_bins > 0, SV_bins / np.maximum(M_bins, 1), 0.0)
    phi_per_test_local = bin_means.mean(axis=2)  # (n_test, k_local)

    # Map back to global training indices
    phi_lc = np.zeros(n_train, dtype=float)
    for t in range(n_test):
        L = neighbors_list[t]
        phi_lc[L] += phi_per_test_local[t, :len(L)]

    save_results(phi_lc, num_samples_done, conv_tracker.convergence_history,
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed)

    return phi_lc, num_samples_done, conv_tracker.convergence_history
