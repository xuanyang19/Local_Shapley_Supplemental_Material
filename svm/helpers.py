import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

warnings.filterwarnings('ignore')


def compute_gamma(X_train):
    """Compute RBF gamma as 1 / (n_features * var(X))."""
    return 1.0 / (X_train.shape[1] * max(X_train.var(), 1e-10))


def precompute_local_neighbors(X_train, X_test, gamma, threshold):
    """For each test point, find training indices with RBF kernel > threshold."""
    neighbors_list = []
    for t in range(len(X_test)):
        K = rbf_kernel(X_train, X_test[t].reshape(1, -1), gamma=gamma).flatten()
        neighbors_list.append(np.where(K > threshold)[0])
    return neighbors_list


def precompute_local_neighbors_with_T_z(X_train, X_test, gamma, threshold):
    """Compute L_t and T_z inverse map. T_z[z] = {t : z in L_t}."""
    n_train = len(X_train)
    L_t = precompute_local_neighbors(X_train, X_test, gamma, threshold)
    T_z = [set() for _ in range(n_train)]
    for t, L in enumerate(L_t):
        for z in L:
            if 0 <= z < n_train:
                T_z[z].add(t)
    return L_t, T_z


def compute_svm_utility(S_indices, X_train, y_train, x_test, y_test_true,
                        gamma, n_classes, seed=42):
    """Train SVM on S_indices and return 1.0 if correct, 0.0 if wrong, 1/C if degenerate."""
    if len(S_indices) == 0:
        return 1.0 / n_classes
    S_indices = list(S_indices)
    y_S = y_train[S_indices]
    if len(S_indices) < 2 or len(np.unique(y_S)) < 2:
        return 1.0 / n_classes
    try:
        clf = SVC(kernel='rbf', gamma=gamma, random_state=seed, cache_size=200)
        clf.fit(X_train[S_indices], y_S)
        pred = clf.predict(x_test.reshape(1, -1))[0]
        return 1.0 if pred == y_test_true else 0.0
    except Exception:
        return 1.0 / n_classes


def compute_svm_utility_batch(S_indices, X_train, y_train, X_test, y_test,
                              test_indices, gamma, n_classes, seed=42):
    """Train SVM on S_indices, batch-predict on multiple test points."""
    if len(S_indices) == 0:
        return np.full(len(test_indices), 1.0 / n_classes)
    S_indices = list(S_indices)
    y_S = y_train[S_indices]
    if len(S_indices) < 2 or len(np.unique(y_S)) < 2:
        return np.full(len(test_indices), 1.0 / n_classes)
    try:
        clf = SVC(kernel='rbf', gamma=gamma, random_state=seed, cache_size=200)
        clf.fit(X_train[S_indices], y_S)
        preds = clf.predict(X_test[test_indices])
        return (preds == y_test[test_indices]).astype(float)
    except Exception:
        return np.full(len(test_indices), 1.0 / n_classes)


def build_log_factorial_table(max_n):
    """Precompute log(i!) for i = 0..max_n. Enables O(1) importance weights."""
    log_fact = np.zeros(max_n + 1)
    for i in range(1, max_n + 1):
        log_fact[i] = log_fact[i - 1] + np.log(i)
    return log_fact


def importance_weight(L_t, L_tp, k, log_fact):
    """O(1) importance weight using precomputed log-factorials.

    w = (L_t+1)/(L_tp+1) * C(L_t, k) / C(L_tp, k)
    """
    if L_t == L_tp or k == 0:
        return 1.0
    # log C(n, k) = log_fact[n] - log_fact[k] - log_fact[n-k]
    log_w = (np.log(L_t + 1) - np.log(L_tp + 1)
             + log_fact[L_t] - log_fact[L_t - k]
             - log_fact[L_tp] + log_fact[L_tp - k])
    return np.exp(log_w)
