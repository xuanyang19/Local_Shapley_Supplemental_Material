#!/usr/bin/env python3
"""
WKNN prediction and distance helpers.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


def _wknn_predict_manual(topk_idx, dist_vec, y_train, n_classes):
    """Distance-weighted KNN prediction from a list of top-K indices."""
    if not topk_idx:
        return -1
    d = dist_vec[topk_idx]
    w = 1.0 / (d + 1e-9)
    labs = y_train[topk_idx]
    class_weights = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        class_weights[c] = np.sum(w[labs == c])
    return int(np.argmax(class_weights))


def _compute_utility_naive(coalition, dist_vec, y_train, y_true, k, n_classes):
    """Utility of a coalition for one test point (global indices)."""
    if len(coalition) == 0:
        return 1.0 / n_classes
    if len(coalition) <= k:
        topk = list(coalition)
    else:
        topk = sorted(coalition, key=lambda idx: dist_vec[idx])[:k]
    pred = _wknn_predict_manual(topk, dist_vec, y_train, n_classes)
    return 1.0 if pred == y_true else 0.0


def _compute_utility_local_naive(coalition_pos, dist_local, y_local, y_true, k, n_classes):
    """Utility of a coalition for one test point (local positions)."""
    if len(coalition_pos) == 0:
        return 1.0 / n_classes
    if len(coalition_pos) <= k:
        topk_pos = list(coalition_pos)
    else:
        topk_pos = sorted(coalition_pos, key=lambda p: dist_local[p])[:k]
    pred = _wknn_predict_manual(topk_pos, dist_local, y_local, n_classes)
    return 1.0 if pred == y_true else 0.0


def precompute_all_distances(X_train, X_test):
    """Pairwise distances: (n_test, n_train)."""
    return pairwise_distances(X_test, X_train, metric='euclidean')


def precompute_knn_neighbors(X_train, X_test, k_local):
    """Precompute k-nearest training neighbors for each test point."""
    nn = NearestNeighbors(n_neighbors=min(k_local, len(X_train)), metric='euclidean')
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_test)
    return indices, distances


def precompute_knn_neighbors_with_T_z(X_train, X_test, k_local):
    """Precompute KNN neighborhoods + T_z inverse map.

    Returns:
        indices, distances: same as precompute_knn_neighbors
        T_z: list of sets, T_z[z] = {t : z in L_t}
    """
    indices, distances = precompute_knn_neighbors(X_train, X_test, k_local)
    n_train = len(X_train)
    T_z = [set() for _ in range(n_train)]
    for t, neighbors in enumerate(indices):
        for z in neighbors:
            if 0 <= z < n_train:
                T_z[z].add(t)
    return indices, distances, T_z


def build_log_factorial_table(max_n):
    """Precompute log(i!) for i = 0..max_n."""
    log_fact = np.zeros(max_n + 1)
    for i in range(1, max_n + 1):
        log_fact[i] = log_fact[i - 1] + np.log(i)
    return log_fact


def importance_weight(L_t_size, L_tp_size, k, log_fact):
    """Importance weight: w = (L_t+1)/(L_tp+1) * C(L_t, k) / C(L_tp, k)."""
    if L_t_size == L_tp_size or k == 0:
        return 1.0
    log_w = (np.log(L_t_size + 1) - np.log(L_tp_size + 1)
             + log_fact[L_t_size] - log_fact[L_t_size - k]
             - log_fact[L_tp_size] + log_fact[L_tp_size - k])
    return np.exp(log_w)
