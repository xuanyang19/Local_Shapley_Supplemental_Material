"""
Abstract interfaces for Shapley value computation.

To use this template:
  1. Subclass ModelWrapper — implement compute_utility(S_indices, test_point_idx)
  2. Subclass LocalRegion — implement get_neighbors(test_idx), n_test, n_train
  3. Pass instances to methods in methods.py

See examples/ for WKNN and SVM implementations.
"""

from abc import ABC, abstractmethod
import numpy as np


class ModelWrapper(ABC):
    """Wraps a model: train on subset S, evaluate on a single test point."""

    @abstractmethod
    def compute_utility(self, S_indices, test_point_idx):
        """Train on S_indices, return utility (e.g. accuracy) for test_point_idx.

        Args:
            S_indices: list/array of training indices to train on.
                       Empty list means no training data (return default utility).
            test_point_idx: index of the test point to evaluate.

        Returns:
            float: utility value (e.g., 1.0 if correct, 0.0 if wrong).
        """
        ...

    @abstractmethod
    def compute_utility_batch(self, S_indices, test_point_indices):
        """Train on S_indices, return utility for multiple test points.

        Args:
            S_indices: list/array of training indices.
            test_point_indices: array of test point indices.

        Returns:
            np.ndarray of shape (len(test_point_indices),): utility per test point.
        """
        ...

    @property
    @abstractmethod
    def n_classes(self):
        """Number of classes."""
        ...

    @property
    def default_utility(self):
        """Utility when S is empty (random baseline)."""
        return 1.0 / self.n_classes


class LocalRegion(ABC):
    """Defines local neighborhoods L_t for each test point t."""

    @abstractmethod
    def get_neighbors(self, test_idx):
        """Return L_t: training indices in local region of test point t.

        Args:
            test_idx: index of the test point.

        Returns:
            np.ndarray of training indices.
        """
        ...

    @property
    @abstractmethod
    def n_test(self):
        """Number of test points."""
        ...

    @property
    @abstractmethod
    def n_train(self):
        """Number of training points."""
        ...

    def get_T_z(self):
        """Compute inverse map T_z[z] = {t : z in L_t}. Used by LSMR-A."""
        T_z = [set() for _ in range(self.n_train)]
        for t in range(self.n_test):
            for z in self.get_neighbors(t):
                if 0 <= z < self.n_train:
                    T_z[z].add(t)
        return T_z
