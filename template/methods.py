"""
Five Shapley value methods using abstract ModelWrapper and LocalRegion interfaces.

  1. Global MC       — full permutation over all training points
  2. TMC-Shapley     — truncated MC with early stopping
  3. Complementary   — paired v(S)-v(Sc), 2D binned estimator
  4. Local MC        — test-centric, permute L_t, marginal contributions
  5. LSMR-A          — local reuse with T_S pivot/skip, single v(S), (L+1) factor
"""

import numpy as np
from .utils import ConvergenceTracker, save_results, maybe_print_progress
from .config import CONVERGENCE_THRESHOLD, CHECK_INTERVAL, MAX_SAMPLES


# ============================================================
# 1. Global MC
# ============================================================
def shapley_global_mc(model, *, max_samples=MAX_SAMPLES, seed=42,
                      n_train, n_test, print_interval=50,
                      out_dir="results/global_mc", dataset="UNK",
                      convergence_threshold=CONVERGENCE_THRESHOLD,
                      check_interval=CHECK_INTERVAL):
    """Full permutation MC Shapley over all training points."""
    tag = "global_mc"
    rng = np.random.default_rng(seed)
    conv_tracker = ConvergenceTracker(n_train, check_interval, convergence_threshold)
    phi_sum = np.zeros(n_train, dtype=float)
    num_samples_done = 0

    for m in range(max_samples):
        contrib = np.zeros(n_train, dtype=float)
        for t in range(n_test):
            pi = rng.permutation(n_train)
            v_prev = model.default_utility
            coalition = []
            for z in pi:
                coalition.append(int(z))
                v_curr = model.compute_utility(coalition, t)
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
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed)
    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ============================================================
# 2. TMC-Shapley
# ============================================================
def shapley_tmc(model, *, max_samples=MAX_SAMPLES, seed=42, patience=5,
                n_train, n_test, print_interval=50,
                out_dir="results/tmc", dataset="UNK",
                convergence_threshold=CONVERGENCE_THRESHOLD,
                check_interval=CHECK_INTERVAL):
    """Truncated MC Shapley with patience-based early stopping per test point."""
    tag = "tmc"
    rng = np.random.default_rng(seed)
    conv_tracker = ConvergenceTracker(n_train, check_interval, convergence_threshold)
    phi_sum = np.zeros(n_train, dtype=float)
    num_samples_done = 0

    # Precompute full-model accuracy per test point
    full_coalition = list(range(n_train))
    full_acc = np.array([model.compute_utility(full_coalition, t) for t in range(n_test)])

    for m in range(max_samples):
        contrib = np.zeros(n_train, dtype=float)
        for t in range(n_test):
            pi = rng.permutation(n_train)
            v_prev = model.default_utility
            coalition = []
            stable_count = 0

            for z in pi:
                coalition.append(int(z))
                v_curr = model.compute_utility(coalition, t)
                contrib[z] += (v_curr - v_prev)
                v_prev = v_curr

                if v_curr == full_acc[t]:
                    stable_count += 1
                    if stable_count >= patience:
                        break
                else:
                    stable_count = 0

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
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed)
    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ============================================================
# 3. Complementary (2D binned)
# ============================================================
def shapley_complementary(model, *, max_samples=MAX_SAMPLES, seed=42,
                          n_train, n_test, print_interval=50,
                          out_dir="results/complementary", dataset="UNK",
                          convergence_threshold=CONVERGENCE_THRESHOLD,
                          check_interval=CHECK_INTERVAL):
    """Complementary pairs v(S)-v(Sc) with 2D binned stratified estimator."""
    tag = "complementary"
    rng = np.random.default_rng(seed)
    conv_tracker = ConvergenceTracker(n_train, check_interval, convergence_threshold)

    SV_bins = np.zeros((n_train, n_train), dtype=float)
    M_bins = np.zeros((n_train, n_train), dtype=int)
    num_samples_done = 0

    for m in range(max_samples):
        i_split = (m % n_train) + 1
        pi = rng.permutation(n_train)
        S, S_comp = pi[:i_split], pi[i_split:]

        total_u = 0.0
        for t in range(n_test):
            v_S = model.compute_utility(S.tolist(), t)
            v_Sc = model.compute_utility(S_comp.tolist(), t) if S_comp.size > 0 else model.default_utility
            total_u += (v_S - v_Sc)

        j_plus = i_split - 1
        j_minus = n_train - i_split - 1

        SV_bins[S, j_plus] += total_u
        M_bins[S, j_plus] += 1
        if S_comp.size > 0 and j_minus >= 0:
            SV_bins[S_comp, j_minus] -= total_u
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
                print(f"\n[{tag}] CONVERGED at sample {num_samples_done}!")
                break
        maybe_print_progress(tag, m, conv_metric, print_interval)

    with np.errstate(divide='ignore', invalid='ignore'):
        means_per_bin = np.where(M_bins > 0, SV_bins / np.maximum(M_bins, 1), 0.0)
    phi_hat = means_per_bin.mean(axis=1)

    save_results(phi_hat, num_samples_done, conv_tracker.convergence_history,
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed)
    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ============================================================
# 4. Local MC (test-centric)
# ============================================================
def shapley_local_mc(model, local_region, *, max_samples=MAX_SAMPLES, seed=42,
                     print_interval=50, out_dir="results/local_mc", dataset="UNK",
                     convergence_threshold=CONVERGENCE_THRESHOLD,
                     check_interval=CHECK_INTERVAL):
    """Test-centric Local MC: permute L_t, compute marginal contributions."""
    tag = "local_mc"
    rng = np.random.default_rng(seed)
    n_train = local_region.n_train
    n_test = local_region.n_test
    conv_tracker = ConvergenceTracker(n_train, check_interval, convergence_threshold)
    phi_sum = np.zeros(n_train, dtype=float)
    num_samples_done = 0

    # Precompute neighbor lists
    L_t = [local_region.get_neighbors(t) for t in range(n_test)]

    for m in range(max_samples):
        contrib = np.zeros(n_train, dtype=float)

        for t in range(n_test):
            L = L_t[t]
            L_len = len(L)
            if L_len == 0:
                continue

            perm = L[rng.permutation(L_len)]
            v_prev = model.default_utility
            coalition = []

            for z in perm:
                coalition.append(int(z))
                v_curr = model.compute_utility(coalition, t)
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
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed)
    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ============================================================
# 5. LSMR-A (local reuse with T_S pivot/skip)
# ============================================================
def shapley_lsmr_a(model, local_region, *, max_samples=MAX_SAMPLES, seed=42,
                   print_interval=50, out_dir="results/lsmr_a", dataset="UNK",
                   convergence_threshold=CONVERGENCE_THRESHOLD,
                   check_interval=CHECK_INTERVAL,
                   use_importance_weight=True,
                   use_reuse=True):
    """LSMR-A (Alg. 1): T_S pivot/skip, importance-weighted redistribution.

    When use_reuse=False, disables T_S pivot/skip and importance weighting.
    Each test point independently samples S from its own L_t, computes a single
    v(S) on that test point, and distributes with the (L+1) factor.
    """
    tag = "lsmr_a"
    rng = np.random.default_rng(seed)
    n_train = local_region.n_train
    n_test = local_region.n_test
    conv_tracker = ConvergenceTracker(n_train, check_interval, convergence_threshold)
    phi_sum = np.zeros(n_train, dtype=float)

    # Precompute L_t and sizes (needed by both paths)
    L_t = [local_region.get_neighbors(t) for t in range(n_test)]
    L_sizes = np.array([len(L) for L in L_t], dtype=int)

    if use_reuse:
        # Precompute T_z, log factorials, ordering (reuse path only)
        T_z = local_region.get_T_z()
        max_L = int(L_sizes.max()) if len(L_sizes) > 0 else 0
        _log_fact = np.zeros(max_L + 2)
        for _i in range(1, max_L + 2):
            _log_fact[_i] = _log_fact[_i - 1] + np.log(_i)

        Pi_T = np.arange(n_test, dtype=int)
        order = {t: i for i, t in enumerate(Pi_T)}
        T_all = set(range(n_test))
        inS_mask = np.zeros(n_train, dtype=bool)

        def _importance_weight(L_t_size, L_tp_size, k):
            if L_t_size == L_tp_size or k == 0:
                return 1.0
            log_w = (np.log(L_t_size + 1) - np.log(L_tp_size + 1)
                     + _log_fact[L_t_size] - _log_fact[L_t_size - k]
                     - _log_fact[L_tp_size] + _log_fact[L_tp_size - k])
            return np.exp(log_w)

    num_samples_done = 0

    for m in range(max_samples):
        delta_phi = np.zeros(n_train, dtype=float)

        if use_reuse:
            # --- Reuse path: T_S pivot/skip, batch eval, importance weights ---
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

                # Train ONCE, evaluate on all T_S test points
                T_S_arr = np.array(sorted(T_S, key=lambda x: order[x]), dtype=int)
                v_vals = model.compute_utility_batch(S.tolist() if S_size > 0 else [], T_S_arr)

                if S_size > 0:
                    inS_mask[S] = True

                for idx, t_prime in enumerate(T_S_arr):
                    L_tp = L_t[t_prime]
                    L_tp_size = int(L_sizes[t_prime])
                    v_tp = v_vals[idx]

                    # Importance weight: corrects sampling from N(t) to N(t')
                    w = _importance_weight(L_len, L_tp_size, S_size) if use_importance_weight else 1.0
                    factor = w * float(L_tp_size + 1) * v_tp

                    if S_size > 0:
                        in_flags = inS_mask[L_tp]
                        L_in, L_out = L_tp[in_flags], L_tp[~in_flags]
                    else:
                        L_in, L_out = np.empty(0, dtype=int), L_tp

                    if S_size > 0 and L_in.size > 0:
                        delta_phi[L_in] += factor / float(S_size)
                    out_den = L_tp_size - S_size
                    if L_out.size > 0 and out_den > 0:
                        delta_phi[L_out] -= factor / float(out_den)

                if S_size > 0:
                    inS_mask[S] = False
        else:
            # --- No-reuse path: independent per test point, no T_S, no importance weight ---
            for t in range(n_test):
                L = L_t[t]
                L_len = len(L)
                if L_len == 0:
                    continue

                i_pos = rng.integers(0, L_len + 1)
                S = rng.choice(L, size=i_pos, replace=False) if i_pos > 0 else np.empty(0, dtype=int)
                S_size = S.size

                v_tp = model.compute_utility(S.tolist() if S_size > 0 else [], t)
                factor = float(L_len + 1) * v_tp

                if S_size > 0:
                    delta_phi[S] += factor / float(S_size)
                out_size = L_len - S_size
                if out_size > 0:
                    S_set = set(S.tolist())
                    L_out = np.array([z for z in L if z not in S_set], dtype=int)
                    delta_phi[L_out] -= factor / float(out_size)

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
                 out_dir=out_dir, tag=tag, dataset=dataset, seed=seed)
    return phi_hat, num_samples_done, conv_tracker.convergence_history
