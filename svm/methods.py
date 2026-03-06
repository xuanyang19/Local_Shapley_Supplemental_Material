#!/usr/bin/env python3
"""Five RBF-SVM Shapley value methods: Global MC, TMC, Complementary, Local MC, LSMR-A."""

import numpy as np
from multiprocessing import Pool

from .helpers import (compute_gamma, compute_svm_utility, compute_svm_utility_batch,
                      precompute_local_neighbors, precompute_local_neighbors_with_T_z,
                      importance_weight, build_log_factorial_table)
from .utils import ConvergenceTracker, save_results, maybe_print_progress


# ================================================================
# 1. Global MC  (parallel)
# ================================================================
def _global_mc_worker(args):
    """Worker: one full permutation sample over all training points."""
    X_tr, y_tr, X_te, y_te, base_seed, m_idx, n_classes, gamma = args
    n_tr = len(X_tr)
    n_te = len(X_te)
    rng = np.random.default_rng(base_seed + m_idx)
    contrib = np.zeros(n_tr, dtype=float)

    for t in range(n_te):
        x_t, y_t = X_te[t], y_te[t]
        pi = rng.permutation(n_tr)
        v_prev = 1.0 / n_classes
        coalition = []

        for z in pi:
            coalition.append(z)
            v_curr = compute_svm_utility(
                coalition, X_tr, y_tr, x_t, y_t, gamma, n_classes, seed=base_seed)
            contrib[z] += (v_curr - v_prev)
            v_prev = v_curr

    return contrib


def shapley_global_mc(X_tr, y_tr, X_te, y_te, *,
                      max_samples, seed, n_jobs, print_interval,
                      out_dir, dataset, convergence_threshold,
                      check_interval, gamma, kernel_threshold):
    tag = "global_mc"
    n_tr = len(X_tr)
    n_classes = len(np.unique(y_tr))
    if gamma is None:
        gamma = compute_gamma(X_tr)

    conv_tracker = ConvergenceTracker(n_tr, check_interval, convergence_threshold)
    phi_sum = np.zeros(n_tr, dtype=float)
    num_samples_done = 0

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} RBF-SVM (parallel, n_jobs={n_jobs})...")
    print(f"{'='*60}")

    args_list = [
        (X_tr, y_tr, X_te, y_te, seed, m, n_classes, gamma)
        for m in range(max_samples)
    ]

    with Pool(processes=n_jobs) as pool:
        for contrib in pool.imap_unordered(_global_mc_worker, args_list):
            phi_sum += contrib
            num_samples_done += 1

            conv_metric = None
            if num_samples_done % check_interval == 0:
                phi_current = phi_sum / float(num_samples_done)
                conv_result = conv_tracker.check_convergence(phi_current, num_samples_done)
                conv_metric = conv_result["convergence_metric"]
                if conv_result["converged"]:
                    print(f"\n[{tag}] CONVERGED at sample {num_samples_done}!")
                    pool.terminate()
                    break

            maybe_print_progress(tag, num_samples_done - 1, conv_metric, print_interval)

    phi_hat = phi_sum / float(num_samples_done)
    save_results(phi_hat, num_samples_done, conv_tracker.convergence_history,
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed,
                 gamma_val=gamma, kernel_threshold=kernel_threshold)
    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ================================================================
# 2. TMC-Shapley  (parallel)
# ================================================================
def _tmc_worker(args):
    """Worker: one TMC sample with early stopping."""
    (X_tr, y_tr, X_te, y_te, yhat_full,
     base_seed, m_idx, patience, n_classes, gamma) = args
    n_tr = len(X_tr)
    n_te = len(X_te)
    rng = np.random.default_rng(base_seed + m_idx)
    contrib = np.zeros(n_tr, dtype=float)

    for t in range(n_te):
        x_t, y_t = X_te[t], y_te[t]
        full_pred = yhat_full[t]
        pi = rng.permutation(n_tr)
        v_prev = 1.0 / n_classes
        coalition = []
        stable_count = 0
        last_pred = None

        for z in pi:
            coalition.append(z)
            v_curr = compute_svm_utility(
                coalition, X_tr, y_tr, x_t, y_t, gamma, n_classes, seed=base_seed)
            contrib[z] += (v_curr - v_prev)
            v_prev = v_curr

            pred = 1 if v_curr == 1.0 else 0  # simplified: correct or not
            # More precise: recheck prediction
            try:
                if len(coalition) >= 2 and len(np.unique(y_tr[coalition])) >= 2:
                    from sklearn.svm import SVC
                    clf = SVC(kernel='rbf', gamma=gamma, random_state=base_seed,
                              cache_size=200)
                    clf.fit(X_tr[coalition], y_tr[coalition])
                    pred = clf.predict(x_t.reshape(1, -1))[0]
                else:
                    pred = None
            except Exception:
                pred = None

            if pred is not None and pred == full_pred:
                stable_count = (stable_count + 1) if last_pred == pred else 1
                if stable_count >= patience and len(coalition) >= 2:
                    break
            else:
                stable_count = 0
            last_pred = pred

    return contrib


def shapley_tmc(X_tr, y_tr, X_te, y_te, *,
                max_samples, seed, patience, n_jobs, print_interval,
                out_dir, dataset, convergence_threshold,
                check_interval, gamma, kernel_threshold):
    tag = "tmc"
    n_tr = len(X_tr)
    n_classes = len(np.unique(y_tr))
    if gamma is None:
        gamma = compute_gamma(X_tr)

    conv_tracker = ConvergenceTracker(n_tr, check_interval, convergence_threshold)
    phi_sum = np.zeros(n_tr, dtype=float)
    num_samples_done = 0

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} RBF-SVM (parallel, n_jobs={n_jobs})...")
    print(f"Precomputing full model predictions...")
    print(f"{'='*60}")

    # Precompute full-model predictions
    yhat_full = np.zeros(len(X_te), dtype=int)
    try:
        from sklearn.svm import SVC
        clf_full = SVC(kernel='rbf', gamma=gamma, random_state=seed, cache_size=200)
        clf_full.fit(X_tr, y_tr)
        yhat_full = clf_full.predict(X_te)
    except Exception:
        # Fallback: majority class
        from collections import Counter
        maj = Counter(y_tr).most_common(1)[0][0]
        yhat_full[:] = maj

    args_list = [
        (X_tr, y_tr, X_te, y_te, yhat_full, seed, m, patience, n_classes, gamma)
        for m in range(max_samples)
    ]

    with Pool(processes=n_jobs) as pool:
        for contrib in pool.imap_unordered(_tmc_worker, args_list):
            phi_sum += contrib
            num_samples_done += 1

            conv_metric = None
            if num_samples_done % check_interval == 0:
                phi_current = phi_sum / float(num_samples_done)
                conv_result = conv_tracker.check_convergence(phi_current, num_samples_done)
                conv_metric = conv_result["convergence_metric"]
                if conv_result["converged"]:
                    print(f"\n[{tag}] CONVERGED at sample {num_samples_done}!")
                    pool.terminate()
                    break

            maybe_print_progress(tag, num_samples_done - 1, conv_metric, print_interval)

    phi_hat = phi_sum / float(num_samples_done)
    save_results(phi_hat, num_samples_done, conv_tracker.convergence_history,
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed,
                 gamma_val=gamma, kernel_threshold=kernel_threshold)
    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ================================================================
# 3. Complementary (parallel)
# ================================================================
def _comp_worker(args):
    """Worker: one complementary sample (v(S) - v(Sc))."""
    X_tr, y_tr, X_te, y_te, base_seed, m_idx, i_split, n_classes, gamma = args
    n_tr = len(X_tr)
    n_te = len(X_te)
    rng = np.random.default_rng(base_seed + m_idx)

    contrib_S = np.zeros(n_tr, dtype=float)
    contrib_Sc = np.zeros(n_tr, dtype=float)
    count_S = np.zeros(n_tr, dtype=int)
    count_Sc = np.zeros(n_tr, dtype=int)

    for t in range(n_te):
        x_t, y_t = X_te[t], y_te[t]
        pi = rng.permutation(n_tr)
        S = pi[:i_split]
        S_comp = pi[i_split:]

        v_S = compute_svm_utility(
            S.tolist(), X_tr, y_tr, x_t, y_t, gamma, n_classes, seed=base_seed)
        v_Sc = compute_svm_utility(
            S_comp.tolist(), X_tr, y_tr, x_t, y_t, gamma, n_classes, seed=base_seed)

        u_t = v_S - v_Sc

        contrib_S[S] += u_t
        count_S[S] += 1
        if S_comp.size > 0:
            contrib_Sc[S_comp] -= u_t
            count_Sc[S_comp] += 1

    return contrib_S, contrib_Sc, count_S, count_Sc, i_split


def shapley_complementary(X_tr, y_tr, X_te, y_te, *,
                          max_samples, seed, n_jobs, print_interval,
                          out_dir, dataset, convergence_threshold,
                          check_interval, gamma, kernel_threshold):
    tag = "complementary"
    n_tr = len(X_tr)
    n_classes = len(np.unique(y_tr))
    if gamma is None:
        gamma = compute_gamma(X_tr)

    conv_tracker = ConvergenceTracker(n_tr, check_interval, convergence_threshold)

    # 2D bins: (n_tr, n_tr) indexed by (training_point, coalition_size)
    SV_bins = np.zeros((n_tr, n_tr), dtype=float)
    M_bins = np.zeros((n_tr, n_tr), dtype=int)
    phi_sum = np.zeros(n_tr, dtype=float)
    num_samples_done = 0

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} RBF-SVM (parallel, n_jobs={n_jobs})...")
    print(f"{'='*60}")

    args_list = [
        (X_tr, y_tr, X_te, y_te, seed, m,
         (m % n_tr) + 1, n_classes, gamma)
        for m in range(max_samples)
    ]

    with Pool(processes=n_jobs) as pool:
        for result in pool.imap_unordered(_comp_worker, args_list):
            contrib_S, contrib_Sc, count_S, count_Sc, i_split = result

            j_plus = i_split - 1
            j_minus = n_tr - i_split - 1

            # Accumulate into bins
            mask_S = count_S > 0
            if j_plus >= 0 and np.any(mask_S):
                SV_bins[mask_S, j_plus] += contrib_S[mask_S]
                M_bins[mask_S, j_plus] += count_S[mask_S]

            mask_Sc = count_Sc > 0
            if j_minus >= 0 and np.any(mask_Sc):
                SV_bins[mask_Sc, j_minus] += contrib_Sc[mask_Sc]
                M_bins[mask_Sc, j_minus] += count_Sc[mask_Sc]

            phi_sum += (contrib_S + contrib_Sc)
            num_samples_done += 1

            conv_metric = None
            if num_samples_done % check_interval == 0:
                # Use binned estimator for convergence
                with np.errstate(divide='ignore', invalid='ignore'):
                    means_now = np.where(
                        M_bins > 0, SV_bins / np.maximum(M_bins, 1).astype(float), 0.0)
                phi_current = means_now.mean(axis=1)
                conv_result = conv_tracker.check_convergence(phi_current, num_samples_done)
                conv_metric = conv_result["convergence_metric"]
                if conv_result["converged"]:
                    print(f"\n[{tag}] CONVERGED at sample {num_samples_done}!")
                    pool.terminate()
                    break

            maybe_print_progress(tag, num_samples_done - 1, conv_metric, print_interval)

    # Final phi: binned stratified estimator
    with np.errstate(divide='ignore', invalid='ignore'):
        means_per_bin = np.where(
            M_bins > 0, SV_bins / np.maximum(M_bins, 1).astype(float), 0.0)
    phi_hat = means_per_bin.mean(axis=1)

    save_results(phi_hat, num_samples_done, conv_tracker.convergence_history,
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed,
                 gamma_val=gamma, kernel_threshold=kernel_threshold)
    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ================================================================
# 4. Local MC  (sequential)
# ================================================================
def shapley_local_mc(X_train, y_train, X_test, y_test, *,
                     max_samples, seed, print_interval,
                     out_dir, dataset, convergence_threshold,
                     check_interval, gamma, kernel_threshold):
    tag = "local_mc"
    n_train, n_test = len(X_train), len(X_test)
    n_classes = len(np.unique(y_train))
    if gamma is None:
        gamma = compute_gamma(X_train)

    conv_tracker = ConvergenceTracker(n_train, check_interval, convergence_threshold)
    neighbors_list = precompute_local_neighbors(X_train, X_test, gamma, kernel_threshold)

    local_sizes = [len(nb) for nb in neighbors_list]
    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} RBF-SVM (sequential)...")
    print(f"Local neighborhood sizes: min={min(local_sizes)}, "
          f"max={max(local_sizes)}, mean={np.mean(local_sizes):.1f}")
    print(f"{'='*60}")

    phi_sum = np.zeros(n_train, dtype=float)
    rng = np.random.default_rng(seed)
    num_samples_done = 0

    for m in range(max_samples):
        contrib = np.zeros(n_train, dtype=float)

        for t in range(n_test):
            L = neighbors_list[t]
            L_len = len(L)
            if L_len == 0:
                continue

            x_t, y_t = X_test[t], y_test[t]
            perm = L[rng.permutation(L_len)]
            v_prev = 1.0 / n_classes
            coalition = []

            for z in perm:
                coalition.append(z)
                v_curr = compute_svm_utility(
                    coalition, X_train, y_train, x_t, y_t, gamma, n_classes, seed=seed)
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
                print(f"\n[{tag}] CONVERGED at sample {num_samples_done}!")
                break

        maybe_print_progress(tag, m, conv_metric, print_interval)

    phi_hat = phi_sum / float(num_samples_done)
    save_results(phi_hat, num_samples_done, conv_tracker.convergence_history,
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed,
                 gamma_val=gamma, kernel_threshold=kernel_threshold)
    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ================================================================
# 5. LSMR-A  (sequential)
# ================================================================
def shapley_lsmr_a(X_train, y_train, X_test, y_test, *,
                   max_samples, seed, print_interval,
                   out_dir, dataset, convergence_threshold,
                   check_interval, gamma, kernel_threshold,
                   use_importance_weight=True,
                   use_reuse=True):
    """LSMR-A (Alg. 1): T_S pivot/skip, importance-weighted redistribution.

    When use_reuse=False, disables T_S pivot/skip and importance weighting.
    Instead, for each test point, samples S from L_t[t], computes a single
    v(S) on that test point, and distributes with (L+1) factor.
    """
    tag = "lsmr_a"
    n_train, n_test = len(X_train), len(X_test)
    n_classes = len(np.unique(y_train))
    if gamma is None:
        gamma = compute_gamma(X_train)

    conv_tracker = ConvergenceTracker(n_train, check_interval, convergence_threshold)

    if use_reuse:
        L_t, T_z = precompute_local_neighbors_with_T_z(X_train, X_test, gamma, kernel_threshold)
        L_sizes = np.array([len(L) for L in L_t], dtype=int)
        log_fact = build_log_factorial_table(int(L_sizes.max()) + 1)

        Pi_T = np.arange(n_test, dtype=int)
        order = {t: i for i, t in enumerate(Pi_T)}
        T_all = set(range(n_test))
        inS_mask = np.zeros(n_train, dtype=bool)
    else:
        L_t = precompute_local_neighbors(X_train, X_test, gamma, kernel_threshold)
        L_sizes = np.array([len(L) for L in L_t], dtype=int)

    local_sizes = L_sizes.tolist()
    mode_str = "T_S pivot/skip, importance-weighted" if use_reuse else "no-reuse, per-test-point"
    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} RBF-SVM ({mode_str})...")
    print(f"Local neighborhood sizes: min={min(local_sizes)}, "
          f"max={max(local_sizes)}, mean={np.mean(local_sizes):.1f}")
    print(f"{'='*60}")

    phi_sum = np.zeros(n_train, dtype=float)
    rng = np.random.default_rng(seed)
    num_samples_done = 0

    for m in range(max_samples):
        delta_phi = np.zeros(n_train, dtype=float)

        if use_reuse:
            # --- Reuse path: T_S pivot/skip, importance-weighted ---
            for t in Pi_T:
                L = L_t[t]
                L_len = len(L)
                if L_len == 0:
                    continue

                i_pos = rng.integers(0, L_len + 1)
                S = rng.choice(L, size=i_pos, replace=False) if i_pos > 0 else np.empty(0, dtype=int)
                S_size = S.size

                # R_S = intersection of T_z for all z in S
                if S_size == 0:
                    T_S = T_all
                else:
                    sets = sorted([T_z[z] for z in S], key=len)
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

                # Train ONCE, batch-predict on all T_S test points
                T_S_arr = np.array(sorted(T_S, key=lambda x: order[x]), dtype=int)
                v_vals = compute_svm_utility_batch(
                    S.tolist(), X_train, y_train, X_test, y_test,
                    T_S_arr, gamma, n_classes, seed=seed)

                if S_size > 0:
                    inS_mask[S] = True

                S_set = set(S.tolist())
                for idx, t_prime in enumerate(T_S_arr):
                    L_tp = L_t[t_prime]
                    L_tp_size = int(L_sizes[t_prime])
                    v_tp = v_vals[idx]

                    # Importance weight: corrects sampling from N(t) to N(t')
                    w = importance_weight(L_len, L_tp_size, S_size, log_fact) if use_importance_weight else 1.0
                    factor = w * float(L_tp_size + 1) * v_tp

                    if S_size > 0:
                        in_flags = inS_mask[L_tp]
                        L_in = L_tp[in_flags]
                        L_out = L_tp[~in_flags]
                    else:
                        L_in = np.empty(0, dtype=int)
                        L_out = L_tp

                    if S_size > 0 and L_in.size > 0:
                        delta_phi[L_in] += factor / float(S_size)
                    out_den = L_tp_size - S_size
                    if L_out.size > 0 and out_den > 0:
                        delta_phi[L_out] -= factor / float(out_den)

                if S_size > 0:
                    inS_mask[S] = False

        else:
            # --- No-reuse path: per-test-point, single v(S) ---
            for t in range(n_test):
                L = L_t[t]
                L_len = len(L)
                if L_len == 0:
                    continue

                x_t, y_t = X_test[t], y_test[t]
                i_pos = rng.integers(0, L_len + 1)
                S = rng.choice(L, size=i_pos, replace=False) if i_pos > 0 else np.empty(0, dtype=int)
                S_size = S.size

                v_tp = compute_svm_utility(
                    S.tolist(), X_train, y_train, x_t, y_t, gamma, n_classes, seed=seed)

                factor = float(L_len + 1) * v_tp

                if S_size > 0:
                    delta_phi[S] += factor / float(S_size)
                out_den = L_len - S_size
                if out_den > 0:
                    non_S = np.setdiff1d(L, S, assume_unique=True)
                    delta_phi[non_S] -= factor / float(out_den)

        phi_sum += delta_phi
        num_samples_done = m + 1

        conv_metric = None
        if num_samples_done % check_interval == 0:
            phi_current = phi_sum / float(num_samples_done)
            conv_result = conv_tracker.check_convergence(phi_current, num_samples_done)
            conv_metric = conv_result["convergence_metric"]
            if conv_result["converged"]:
                print(f"\n[{tag}] CONVERGED at sample {num_samples_done}!")
                break

        maybe_print_progress(tag, m, conv_metric, print_interval)

    phi_hat = phi_sum / float(num_samples_done)
    save_results(phi_hat, num_samples_done, conv_tracker.convergence_history,
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed,
                 gamma_val=gamma, kernel_threshold=kernel_threshold)
    return phi_hat, num_samples_done, conv_tracker.convergence_history
