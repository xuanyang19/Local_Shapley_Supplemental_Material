"""
Example: RBF-SVM model + kernel-based local region.

Usage:
    python -m template.run --model-module template.examples.svm_model --dataset Iris --methods global_mc lsmr_a
"""

import numpy as np
from sklearn.svm import SVC
from template.model_interface import ModelWrapper, LocalRegion


KERNEL_THRESHOLD = 0.01


class SVMModel(ModelWrapper):
    """RBF-SVM classifier."""

    def __init__(self, X_train, y_train, X_test, y_test, gamma, seed=42):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._gamma = gamma
        self._seed = seed
        self._n_classes = len(np.unique(y_train))

    @property
    def n_classes(self):
        return self._n_classes

    def compute_utility(self, S_indices, test_point_idx):
        if len(S_indices) == 0:
            return self.default_utility
        X_S = self._X_train[S_indices]
        y_S = self._y_train[S_indices]
        if len(np.unique(y_S)) < 2:
            pred = y_S[0]
        else:
            clf = SVC(kernel='rbf', gamma=self._gamma, random_state=self._seed)
            clf.fit(X_S, y_S)
            pred = clf.predict(self._X_test[test_point_idx:test_point_idx+1])[0]
        return 1.0 if pred == self._y_test[test_point_idx] else 0.0

    def compute_utility_batch(self, S_indices, test_point_indices):
        if len(S_indices) == 0:
            return np.full(len(test_point_indices), self.default_utility)
        X_S = self._X_train[S_indices]
        y_S = self._y_train[S_indices]
        if len(np.unique(y_S)) < 2:
            preds = np.full(len(test_point_indices), y_S[0])
        else:
            clf = SVC(kernel='rbf', gamma=self._gamma, random_state=self._seed)
            clf.fit(X_S, y_S)
            preds = clf.predict(self._X_test[test_point_indices])
        return (preds == self._y_test[test_point_indices]).astype(float)


class KernelLocalRegion(LocalRegion):
    """RBF kernel-based local region: L_t = training points with K(x_t, x_z) > threshold."""

    def __init__(self, X_train, X_test, gamma, threshold=KERNEL_THRESHOLD):
        self._n_train = len(X_train)
        self._n_test = len(X_test)
        self._neighbors = []
        for t in range(self._n_test):
            dists_sq = np.sum((X_train - X_test[t]) ** 2, axis=1)
            kernel_vals = np.exp(-gamma * dists_sq)
            self._neighbors.append(np.where(kernel_vals > threshold)[0].astype(int))

    def get_neighbors(self, test_idx):
        return self._neighbors[test_idx]

    @property
    def n_test(self):
        return self._n_test

    @property
    def n_train(self):
        return self._n_train


def setup_model(args, seed):
    """Called by template/run.py. Returns (model, local_region, n_train, n_test)."""
    from data import load_data
    X_train, y_train, X_test, y_test = load_data(
        args.dataset, sample_size=tuple(args.sample_size), random_state=seed)

    gamma = 1.0 / (X_train.shape[1] * X_train.var())
    model = SVMModel(X_train, y_train, X_test, y_test, gamma, seed=seed)
    local_region = KernelLocalRegion(X_train, X_test, gamma, threshold=KERNEL_THRESHOLD)
    return model, local_region, len(X_train), len(X_test)
