#!/usr/bin/env python3
"""Multiprocessing worker functions for GNN Shapley methods."""

import numpy as np
import torch
from math import lgamma

from .model import train_and_eval_gnn

# ================================================================
# IPC globals (set by init_worker)
# ================================================================
g_cache = None
g_data_np = None


def init_worker(cache, data_np):
    global g_cache, g_data_np
    g_cache = cache
    g_data_np = data_np


def _get_gpu_device(worker_id):
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        return torch.device(f"cuda:{worker_id % gpu_count}")
    return torch.device("cpu")


# ================================================================
# 1. Global MC
# ================================================================
def _worker_global_mc(args):
    (sample_idx, n_classes, base_seed, epochs) = args
    device = _get_gpu_device(sample_idx)
    rng = np.random.default_rng(base_seed + sample_idx * 9973)

    n_train = g_cache.n_train
    contrib = np.zeros(n_train, dtype=float)
    test_nodes = g_data_np['test_indices']
    n_test = len(test_nodes)
    n_evals = 0

    for t_i in range(n_test):
        test_node = test_nodes[t_i]
        perm = g_cache.train_list[rng.permutation(n_train)]
        v_prev = 1.0 / n_classes
        model_seed = base_seed + sample_idx * 10000 + t_i

        for i, z in enumerate(perm):
            S = perm[:i + 1]
            acc_vec = train_and_eval_gnn(
                S, g_data_np['x'], g_data_np['y'], g_data_np['edge_index'],
                np.array([test_node]), n_classes, model_seed, epochs, device)
            n_evals += 1
            v_curr = acc_vec[0]
            z_idx = g_cache.node_to_idx[int(z)]
            contrib[z_idx] += (v_curr - v_prev)
            v_prev = v_curr

    return contrib, n_evals


# ================================================================
# 2. TMC-Shapley
# ================================================================
def _worker_tmc(args):
    (sample_idx, n_classes, base_seed, epochs, v_full_vec, patience) = args
    device = _get_gpu_device(sample_idx)
    rng = np.random.default_rng(base_seed + sample_idx * 9973)

    n_train = g_cache.n_train
    contrib = np.zeros(n_train, dtype=float)
    n_test = len(g_data_np['test_indices'])
    n_evals = 0

    for t_i in range(n_test):
        test_node = g_data_np['test_indices'][t_i]
        full_acc = v_full_vec[t_i]

        perm = g_cache.train_list[rng.permutation(n_train)]
        v_prev = 1.0 / n_classes
        stable_count = 0
        last_acc = None
        model_seed = base_seed + sample_idx * 10000 + t_i

        for i, z in enumerate(perm):
            S = perm[:i + 1]
            acc_vec = train_and_eval_gnn(
                S, g_data_np['x'], g_data_np['y'], g_data_np['edge_index'],
                np.array([test_node]), n_classes, model_seed, epochs, device)
            n_evals += 1
            v_curr = acc_vec[0]

            diff = v_curr - v_prev
            z_idx = g_cache.node_to_idx[int(z)]
            contrib[z_idx] += diff
            v_prev = v_curr

            if v_curr == full_acc:
                stable_count = (stable_count + 1) if last_acc == v_curr else 1
                if stable_count >= patience:
                    break
            else:
                stable_count = 0
            last_acc = v_curr

    return contrib, n_evals


# ================================================================
# 3. Complementary (Global)
# ================================================================
def _worker_complementary(args):
    (sample_idx, i_pos, n_classes, base_seed, epochs) = args
    device = _get_gpu_device(sample_idx)
    rng = np.random.default_rng(base_seed + sample_idx * 9973)

    n_train = g_cache.n_train
    n_evals = 0

    perm = g_cache.train_list[rng.permutation(n_train)]
    S = perm[:i_pos]
    S_comp = perm[i_pos:]

    model_seed = base_seed + sample_idx * 10000

    v_S_vec = train_and_eval_gnn(
        S, g_data_np['x'], g_data_np['y'], g_data_np['edge_index'],
        g_data_np['test_indices'], n_classes, model_seed, epochs, device)
    n_evals += 1
    v_S_sum = np.sum(v_S_vec)

    if len(S_comp) == 0:
        v_Sc_sum = len(g_data_np['test_indices']) * (1.0 / n_classes)
    else:
        v_Sc_vec = train_and_eval_gnn(
            S_comp, g_data_np['x'], g_data_np['y'], g_data_np['edge_index'],
            g_data_np['test_indices'], n_classes, model_seed, epochs, device)
        n_evals += 1
        v_Sc_sum = np.sum(v_Sc_vec)

    total_u = v_S_sum - v_Sc_sum
    perm_indices = np.array([g_cache.node_to_idx[int(z)] for z in perm])

    return perm_indices, i_pos, total_u, n_evals


# ================================================================
# 4. Local MC 
# ================================================================
def _worker_local_mc(args):
    (sample_idx, n_classes, base_seed, epochs) = args
    device = _get_gpu_device(sample_idx)
    rng = np.random.default_rng(base_seed + sample_idx * 9973)

    n_train = g_cache.n_train
    contrib = np.zeros(n_train, dtype=float)
    n_evals = 0
    train_list_np = g_cache.train_list

    for z_idx in range(n_train):
        L_z = g_cache.L_train[z_idx]
        if len(L_z) == 0:
            continue

        T_z_indices = list(g_cache.T_z[z_idx])
        if len(T_z_indices) == 0:
            continue

        actual_test_nodes = g_data_np['test_indices'][T_z_indices]
        z_node = train_list_np[z_idx]

        perm = L_z[rng.permutation(len(L_z))]

        z_pos_arr = np.where(perm == z_node)[0]
        if len(z_pos_arr) == 0:
            continue
        z_pos = z_pos_arr[0]

        S_without_z = perm[:z_pos].tolist()
        S_with_z = perm[:z_pos + 1].tolist()

        model_seed = base_seed + sample_idx * 10000 + z_idx

        if len(S_without_z) == 0:
            v_without = np.full(len(T_z_indices), 1.0 / n_classes)
        else:
            v_without = train_and_eval_gnn(
                S_without_z, g_data_np['x'], g_data_np['y'],
                g_data_np['edge_index'], actual_test_nodes,
                n_classes, model_seed, epochs, device)
            n_evals += 1

        if len(S_with_z) == 0:
            v_with = np.full(len(T_z_indices), 1.0 / n_classes)
        else:
            v_with = train_and_eval_gnn(
                S_with_z, g_data_np['x'], g_data_np['y'],
                g_data_np['edge_index'], actual_test_nodes,
                n_classes, model_seed, epochs, device)
            n_evals += 1

        diff = np.sum(v_with - v_without)
        contrib[z_idx] += diff

    return contrib, n_evals


# ================================================================
# 5. LSMR-A 
# ================================================================
def _worker_lsmr_a(args):
    """Train-centric complementary: v(S)-v(Sc) on local test nodes."""
    (sample_idx, n_classes, base_seed, epochs, use_reuse) = args
    device = _get_gpu_device(sample_idx)
    rng = np.random.default_rng(base_seed + sample_idx * 9973)

    n_train = g_cache.n_train
    n_evals = 0
    test_indices = g_data_np['test_indices']
    node_to_idx = g_cache.node_to_idx

    results = []  # list of (S_indices, Sc_indices, i_pos, L_len, u)

    for z_idx in range(n_train):
        L_z = g_cache.L_train[z_idx]
        if len(L_z) == 0:
            continue

        T_z_set = g_cache.T_z[z_idx]
        if len(T_z_set) == 0:
            continue

        L_len = len(L_z)
        perm = L_z[rng.permutation(L_len)]
        i_pos = int(rng.integers(0, L_len + 1))

        S = perm[:i_pos].tolist() if i_pos > 0 else []
        Sc = perm[i_pos:].tolist() if i_pos < L_len else []

        model_seed = base_seed + sample_idx * 10000 + z_idx

        if not use_reuse or i_pos == 0:
            # No reuse: compute v(S) and v(Sc) for this z_idx only
            T_z_list = list(T_z_set)
            actual_test = test_indices[T_z_list]

            if len(S) == 0:
                v_S = np.full(len(T_z_list), 1.0 / n_classes)
            else:
                v_S = train_and_eval_gnn(
                    S, g_data_np['x'], g_data_np['y'], g_data_np['edge_index'],
                    actual_test, n_classes, model_seed, epochs, device)
                n_evals += 1

            if len(Sc) == 0:
                v_Sc = np.full(len(T_z_list), 1.0 / n_classes)
            else:
                v_Sc = train_and_eval_gnn(
                    Sc, g_data_np['x'], g_data_np['y'], g_data_np['edge_index'],
                    actual_test, n_classes, model_seed, epochs, device)
                n_evals += 1

            u = float(np.sum(v_S - v_Sc))
            S_idx = [node_to_idx[int(z)] for z in S]
            Sc_idx = [node_to_idx[int(z)] for z in Sc]
            results.append((S_idx, Sc_idx, i_pos, L_len, u))

        else:
            # Reuse: R_S = {z' : S subset of L_train[z']}
            S_node_indices = [node_to_idx[int(z)] for z in S]
            R_S = set(g_cache.N_z[S_node_indices[0]])
            for si in S_node_indices[1:]:
                R_S &= g_cache.N_z[si]

            if not R_S or z_idx != min(R_S):
                continue

            # Train v(S) once for union of all T_z' test nodes
            all_test_set = set()
            for zp_idx in R_S:
                all_test_set.update(g_cache.T_z[zp_idx])
            all_test_list = sorted(all_test_set)

            if not all_test_list:
                continue

            actual_test_all = test_indices[all_test_list]
            v_S_all = train_and_eval_gnn(
                S, g_data_np['x'], g_data_np['y'], g_data_np['edge_index'],
                actual_test_all, n_classes, model_seed, epochs, device)
            n_evals += 1

            test_pos_map = {t: i for i, t in enumerate(all_test_list)}
            S_set = set(int(z) for z in S)

            for zp_idx in R_S:
                L_zp = g_cache.L_train[zp_idx]
                L_zp_len = len(L_zp)
                T_zp_list = list(g_cache.T_z[zp_idx])
                if not T_zp_list:
                    continue

                # Sc for z' = L_z' \ S
                Sc_zp = [int(z) for z in L_zp if int(z) not in S_set]

                # v_S for T_z' test points (reuse from v_S_all)
                T_zp_positions = [test_pos_map[t] for t in T_zp_list]
                v_S_zp = v_S_all[T_zp_positions]

                # v_Sc for z'
                actual_test_zp = test_indices[T_zp_list]
                if len(Sc_zp) == 0:
                    v_Sc_zp = np.full(len(T_zp_list), 1.0 / n_classes)
                else:
                    v_Sc_zp = train_and_eval_gnn(
                        Sc_zp, g_data_np['x'], g_data_np['y'],
                        g_data_np['edge_index'], actual_test_zp,
                        n_classes, model_seed + zp_idx, epochs, device)
                    n_evals += 1

                u_zp = float(np.sum(v_S_zp - v_Sc_zp))

                # Importance weight: (L+1)/(L'+1) * C(L, i_pos) / C(L', i_pos)
                if L_len != L_zp_len:
                    log_w = (np.log(L_len + 1) - np.log(L_zp_len + 1)
                             + lgamma(L_len + 1) - lgamma(L_len - i_pos + 1)
                             - lgamma(L_zp_len + 1) + lgamma(L_zp_len - i_pos + 1))
                    u_zp *= np.exp(log_w)

                S_idx = S_node_indices[:]
                Sc_zp_idx = [node_to_idx[int(z)] for z in Sc_zp]
                results.append((S_idx, Sc_zp_idx, i_pos, L_zp_len, u_zp))

    return results, n_evals
