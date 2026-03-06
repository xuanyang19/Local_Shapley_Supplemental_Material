import numpy as np
import local_decision_tree


def get_local_neighbors(X_train, y_train, X_test, n_train, seed):
    """Return L_t: list of arrays, L_t[t] = training indices in local region of test point t."""
    Local_raw = local_decision_tree.precompute_local_neighbors_strict(
        X_train, y_train, X_test,
        neighbor_method="leaf", leaf_mode="single",
        desired_min=30, desired_max=50, knn_k=40, seed=seed,
        tree_kwargs={"min_samples_leaf": 5, "min_samples_split": 10},
    )
    neighbors_list = []
    for t in range(len(X_test)):
        L = np.asarray(Local_raw[t], dtype=int).reshape(-1)
        L = L[(L >= 0) & (L < n_train)]
        neighbors_list.append(L)
    return neighbors_list


def get_local_neighbors_with_T_z(X_train, y_train, X_test, n_train, seed):
    """Compute L_t and T_z inverse map. T_z[z] = {t : z in L_t}."""
    L_t = get_local_neighbors(X_train, y_train, X_test, n_train, seed)
    T_z = [set() for _ in range(n_train)]
    for t, L in enumerate(L_t):
        for z in L:
            if 0 <= z < n_train:
                T_z[z].add(t)
    return L_t, T_z
