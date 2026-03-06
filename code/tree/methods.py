import numpy as np
from multiprocessing import Pool
from sklearn.tree import DecisionTreeClassifier

from .utils import ConvergenceTracker, save_results, maybe_print_progress
from .helpers import get_local_neighbors, get_local_neighbors_with_T_z
from .config import CONVERGENCE_THRESHOLD, CHECK_INTERVAL, MAX_SAMPLES


# ============================================================
# 1. GLOBAL MC (parallel)
# ============================================================
def _global_worker(args):
    (X_tr, y_tr, X_te, y_te, base_seed, m_idx, n_classes) = args
    rng = np.random.default_rng(base_seed + 10007 * m_idx)
    n_tr, n_te = X_tr.shape[0], X_te.shape[0]
    contrib = np.zeros(n_tr, dtype=float)

    for t in range(n_te):
        x_t, y_t = X_te[t:t+1], y_te[t]
        pi = rng.permutation(n_tr).astype(int)
        v_prev = 1.0 / n_classes

        for i, z in enumerate(pi):
            S_plus = pi[:i+1]
            clf = DecisionTreeClassifier(random_state=base_seed)
            clf.fit(X_tr[S_plus], y_tr[S_plus])
            v_curr = float(clf.predict(x_t)[0] == y_t)
            contrib[z] += (v_curr - v_prev)
            v_prev = v_curr
    return contrib


def shapley_global_mc(X_tr, y_tr, X_te, y_te, *,
                      max_samples=MAX_SAMPLES, seed=42, n_jobs=64,
                      print_interval=50, out_dir="tree_results", dataset="UNK",
                      convergence_threshold=CONVERGENCE_THRESHOLD,
                      check_interval=CHECK_INTERVAL):
    tag = "global_mc"
    n_tr = len(X_tr)
    n_classes = len(np.unique(y_tr))
    conv_tracker = ConvergenceTracker(n_tr, check_interval, convergence_threshold)
    phi_sum = np.zeros(n_tr, dtype=float)

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} (n_jobs={n_jobs})...")
    print(f"{'='*60}")

    task_args = [(X_tr, y_tr, X_te, y_te, seed, m, n_classes) for m in range(max_samples)]

    with Pool(processes=n_jobs) as pool:
        num_samples_done = 0

        for m, contrib in enumerate(pool.imap_unordered(_global_worker, task_args)):
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
                    pool.terminate()
                    break
            maybe_print_progress(tag, m, conv_metric, print_interval)

    phi_hat = phi_sum / float(num_samples_done)
    save_results(phi_hat, num_samples_done,
                 conv_tracker.convergence_history, out_dir=out_dir, tag=tag,
                 dataset=dataset, seed=seed)
    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ============================================================
# 2. TMC-Shapley (parallel)
# ============================================================
def _tmc_worker(args):
    (X_tr, y_tr, X_te, y_te, yhat_full, base_seed, m_idx, patience, n_classes) = args
    n_tr, n_te = X_tr.shape[0], X_te.shape[0]
    contrib = np.zeros(n_tr, dtype=float)
    rng = np.random.default_rng(base_seed + 10007 * m_idx)

    for t in range(n_te):
        x_t, y_t = X_te[t:t+1], y_te[t]
        full_pred = yhat_full[t]
        pi = rng.permutation(n_tr).astype(int)
        v_prev = 1.0 / n_classes
        stable_count = 0
        last_pred = None

        for i, z in enumerate(pi):
            S_plus = pi[:i+1]
            clf = DecisionTreeClassifier(random_state=base_seed)
            clf.fit(X_tr[S_plus], y_tr[S_plus])
            pred = clf.predict(x_t)[0]
            v_curr = 1.0 if pred == y_t else 0.0
            contrib[z] += (v_curr - v_prev)
            v_prev = v_curr

            if pred == full_pred:
                stable_count = (stable_count + 1) if last_pred == pred else 1
                if stable_count >= patience:
                    break
            else:
                stable_count = 0
            last_pred = pred
    return contrib


def shapley_tmc(X_tr, y_tr, X_te, y_te, *,
                max_samples=MAX_SAMPLES, seed=42, n_jobs=64, patience=2,
                print_interval=50, out_dir="tree_results", dataset="UNK",
                convergence_threshold=CONVERGENCE_THRESHOLD,
                check_interval=CHECK_INTERVAL):
    tag = "tmc"
    n_tr = X_tr.shape[0]
    n_classes = len(np.unique(y_tr))
    conv_tracker = ConvergenceTracker(n_tr, check_interval, convergence_threshold)
    phi_sum = np.zeros(n_tr, dtype=float)

    clf_full = DecisionTreeClassifier(random_state=seed)
    clf_full.fit(X_tr, y_tr)
    yhat_full = clf_full.predict(X_te)

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} (n_jobs={n_jobs}, patience={patience})...")
    print(f"{'='*60}")

    task_args = [(X_tr, y_tr, X_te, y_te, yhat_full, seed, m, patience, n_classes)
                 for m in range(max_samples)]

    with Pool(processes=n_jobs) as pool:
        num_samples_done = 0

        for m, contrib in enumerate(pool.imap_unordered(_tmc_worker, task_args)):
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
                    pool.terminate()
                    break
            maybe_print_progress(tag, m, conv_metric, print_interval)

    phi_hat = phi_sum / float(num_samples_done)
    save_results(phi_hat, num_samples_done,
                 conv_tracker.convergence_history, out_dir=out_dir, tag=tag,
                 dataset=dataset, seed=seed)
    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ============================================================
# 3. Comple-S (parallel, binned)
# ============================================================
def _comp_worker(args):
    (X_tr, y_tr, X_te, y_te, base_seed, m_idx, i_split, n_classes) = args
    rng = np.random.default_rng(base_seed + 10007 * m_idx)
    n_tr, n_te = X_tr.shape[0], X_te.shape[0]

    pi = rng.permutation(n_tr).astype(int)
    i = i_split
    S, S_comp = pi[:i], pi[i:]

    clf_S = DecisionTreeClassifier(random_state=base_seed)
    clf_S.fit(X_tr[S], y_tr[S])
    v_S_sum = np.sum(clf_S.predict(X_te) == y_te)

    if S_comp.size == 0:
        v_Sc_sum = n_te * (1.0 / n_classes)
    else:
        clf_Sc = DecisionTreeClassifier(random_state=base_seed)
        clf_Sc.fit(X_tr[S_comp], y_tr[S_comp])
        v_Sc_sum = np.sum(clf_Sc.predict(X_te) == y_te)

    return pi, i, v_S_sum - v_Sc_sum


def shapley_complementary(X_tr, y_tr, X_te, y_te, *,
                          max_samples=MAX_SAMPLES, seed=42, n_jobs=64,
                          print_interval=50, out_dir="tree_results", dataset="UNK",
                          convergence_threshold=CONVERGENCE_THRESHOLD,
                          check_interval=CHECK_INTERVAL):
    tag = "complementary"
    n_tr = X_tr.shape[0]
    n_classes = len(np.unique(y_tr))
    conv_tracker = ConvergenceTracker(n_tr, check_interval, convergence_threshold)

    SV_bins = np.zeros((n_tr, n_tr), dtype=float)
    M_bins = np.zeros((n_tr, n_tr), dtype=int)

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} (n_jobs={n_jobs})...")
    print(f"{'='*60}")

    task_args = [(X_tr, y_tr, X_te, y_te, seed, m, (m % n_tr) + 1, n_classes) for m in range(max_samples)]

    with Pool(processes=n_jobs) as pool:
        num_samples_done = 0

        for m, result in enumerate(pool.imap_unordered(_comp_worker, task_args)):
            pi, i, total_u = result
            S, S_comp = pi[:i], pi[i:]
            j_plus, j_minus = i - 1, n_tr - i - 1

            SV_bins[S, j_plus] += total_u
            M_bins[S, j_plus] += 1
            if S_comp.size > 0 and j_minus >= 0:
                SV_bins[S_comp, j_minus] -= total_u
                M_bins[S_comp, j_minus] += 1

            num_samples_done = m + 1

            conv_metric = None
            if num_samples_done % check_interval == 0:
                with np.errstate(divide='ignore', invalid='ignore'):
                    means_per_bin = np.where(M_bins > 0, SV_bins / np.maximum(M_bins, 1), 0.0)
                phi_current = means_per_bin.mean(axis=1)
                conv_result = conv_tracker.check_convergence(phi_current, num_samples_done)
                conv_metric = conv_result["convergence_metric"]
                if conv_result["converged"]:
                    print(f"\n[{tag}] CONVERGED at sample {num_samples_done}! "
                          f"metric={conv_metric:.6f} < {convergence_threshold}")
                    pool.terminate()
                    break
            maybe_print_progress(tag, m, conv_metric, print_interval)

    with np.errstate(divide='ignore', invalid='ignore'):
        means_per_bin = np.where(M_bins > 0, SV_bins / np.maximum(M_bins, 1), 0.0)
    phi_cc = means_per_bin.mean(axis=1)

    save_results(phi_cc, num_samples_done,
                 conv_tracker.convergence_history, out_dir=out_dir, tag=tag,
                 dataset=dataset, seed=seed)
    return phi_cc, num_samples_done, conv_tracker.convergence_history


# ============================================================
# 4. LOCAL MC (sequential)
# ============================================================
def shapley_local_mc(X_train, y_train, X_test, y_test, *,
                     max_samples=MAX_SAMPLES, seed=42, print_interval=50,
                     out_dir="tree_results", dataset="UNK",
                     convergence_threshold=CONVERGENCE_THRESHOLD,
                     check_interval=CHECK_INTERVAL):
    tag = "local_mc"
    rng = np.random.default_rng(seed)
    n_train, n_test = len(X_train), len(X_test)
    n_classes = len(np.unique(y_train))
    conv_tracker = ConvergenceTracker(n_train, check_interval, convergence_threshold)
    phi_sum = np.zeros(n_train, dtype=float)

    neighbors_list = get_local_neighbors(X_train, y_train, X_test, n_train, seed)

    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} (sequential)...")
    print(f"{'='*60}")

    num_samples_done = 0

    for m in range(max_samples):
        contrib = np.zeros(n_train, dtype=float)

        for t in range(n_test):
            L = neighbors_list[t]
            if L.size == 0:
                continue
            y_t, x_t = y_test[t], X_test[t:t+1]
            perm = L[rng.permutation(L.size)]
            acc_prev = 1.0 / n_classes

            for i, z in enumerate(perm):
                S = perm[:i+1]
                clf = DecisionTreeClassifier(random_state=seed)
                clf.fit(X_train[S], y_train[S])
                acc_curr = float(clf.predict(x_t)[0] == y_t)
                contrib[z] += (acc_curr - acc_prev)
                acc_prev = acc_curr

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
    save_results(phi_hat, num_samples_done,
                 conv_tracker.convergence_history, out_dir=out_dir, tag=tag,
                 dataset=dataset, seed=seed)
    return phi_hat, num_samples_done, conv_tracker.convergence_history


# ============================================================
# 5. LSMR-A (local reuse with T_S pivot/skip, sequential)
# ============================================================
def shapley_lsmr_a(X_train, y_train, X_test, y_test, *,
                   max_samples=MAX_SAMPLES, seed=42, print_interval=50,
                   out_dir="tree_results", dataset="UNK",
                   convergence_threshold=CONVERGENCE_THRESHOLD,
                   check_interval=CHECK_INTERVAL,
                   use_importance_weight=True,
                   use_reuse=True):
    """LSMR-A (Alg. 1): T_S pivot/skip, importance-weighted redistribution.

    When use_reuse=False, disables T_S reuse/skip: samples S per test point,
    evaluates a single v(S) on that test point, and distributes with (L+1) factor.
    """
    tag = "lsmr_a"
    rng = np.random.default_rng(seed)
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    n_classes = len(np.unique(y_train))
    conv_tracker = ConvergenceTracker(n_train, check_interval, convergence_threshold)
    phi_sum = np.zeros(n_train, dtype=float)

    if use_reuse:
        L_t, T_z = get_local_neighbors_with_T_z(X_train, y_train, X_test, n_train, seed)
    else:
        L_t = get_local_neighbors(X_train, y_train, X_test, n_train, seed)

    L_sizes = np.array([len(L) for L in L_t], dtype=int)
    max_L = int(L_sizes.max()) if len(L_sizes) > 0 else 0

    if use_reuse:
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

        def utility_vec(S, test_indices):
            """Train on S, batch-predict on test_indices."""
            if len(S) == 0:
                return np.full(len(test_indices), 1.0 / n_classes)
            clf = DecisionTreeClassifier(random_state=seed)
            clf.fit(X_train[S], y_train[S])
            return (clf.predict(X_test[test_indices]) == y_test[test_indices]).astype(float)

    reuse_label = "T_S pivot/skip, importance-weighted" if use_reuse else "no reuse"
    print(f"\n{'='*60}")
    print(f"Running {tag.upper()} ({reuse_label})...")
    print(f"{'='*60}")

    num_samples_done = 0

    for m in range(max_samples):
        delta_phi = np.zeros(n_train, dtype=float)

        if use_reuse:
            # ---------- reuse path (original) ----------
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
                v_vals = utility_vec(S, T_S_arr)

                if S_size > 0:
                    inS_mask[S] = True

                for idx, t_prime in enumerate(T_S_arr):
                    L_tp = L_t[t_prime]
                    L_tp_size = int(L_sizes[t_prime])
                    v_tp = v_vals[idx]

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
            # ---------- no-reuse path ----------
            for t in range(n_test):
                L = L_t[t]
                L_len = len(L)
                if L_len == 0:
                    continue

                i_pos = rng.integers(0, L_len + 1)
                S = rng.choice(L, size=i_pos, replace=False) if i_pos > 0 else np.empty(0, dtype=int)
                S_size = S.size

                # Single v(S) on this test point
                if S_size == 0:
                    v_tp = 1.0 / n_classes
                else:
                    clf = DecisionTreeClassifier(random_state=seed)
                    clf.fit(X_train[S], y_train[S])
                    v_tp = float(clf.predict(X_test[t:t+1])[0] == y_test[t])

                factor = float(L_len + 1) * v_tp

                # S members get +factor/|S|
                if S_size > 0:
                    delta_phi[S] += factor / float(S_size)
                # Non-S members (L\S) get -factor/(L-|S|)
                out_den = L_len - S_size
                if out_den > 0:
                    L_set = set(L)
                    S_set = set(S)
                    L_out = np.array([z for z in L if z not in S_set], dtype=int)
                    delta_phi[L_out] -= factor / float(out_den)

        phi_sum += delta_phi
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
    save_results(phi_hat, num_samples_done,
                 conv_tracker.convergence_history, out_dir=out_dir, tag=tag,
                 dataset=dataset, seed=seed)
    return phi_hat, num_samples_done, conv_tracker.convergence_history
