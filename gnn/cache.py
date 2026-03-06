#!/usr/bin/env python3
"""
LocalRegionCache: precompute k-hop neighborhoods for train and test nodes.

Stores:
  - L_train[z_idx]: train-node neighbourhood (array of node IDs)
  - L_test[t_i]:    test-node neighbourhood  (array of node IDs)
  - T_z[z_idx]:     set of test-point indices that include train node z
  - node_to_idx:    mapping from node ID -> index in train_list
"""

import numpy as np
import time

from .model import get_k_hop_neighbors


class LocalRegionCache:
    def __init__(self, edge_index, train_list, test_indices, k_hop=2, build_N_z=False):
        print("Precomputing local regions...")
        t0 = time.time()

        self.train_list = train_list
        self.n_train = len(train_list)
        self.test_indices = test_indices
        self.n_test = len(test_indices)

        train_set = set(train_list.tolist())
        self.node_to_idx = {int(n): i for i, n in enumerate(self.train_list)}

        # Train-node neighbourhoods
        self.L_train = []
        for z in self.train_list:
            neighbors = get_k_hop_neighbors(int(z), k_hop, edge_index)
            L_z = np.array([int(n) for n in neighbors.numpy()
                            if int(n) in train_set], dtype=int)
            self.L_train.append(L_z)

        self.max_L = max(len(L) for L in self.L_train) if self.L_train else 0

        # Test-node neighbourhoods + inverse map T_z
        self.L_test = []
        self.T_z = [set() for _ in range(self.n_train)]

        for t_i, t_node in enumerate(self.test_indices):
            neighbors = get_k_hop_neighbors(int(t_node), k_hop, edge_index)
            L_t = np.array([int(n) for n in neighbors.numpy()
                            if int(n) in train_set], dtype=int)
            self.L_test.append(L_t)

            for z in L_t:
                if int(z) in self.node_to_idx:
                    z_idx = self.node_to_idx[int(z)]
                    self.T_z[z_idx].add(t_i)

        # N_z[z_idx] = {z_idx' : z is in L_train[z_idx']} (for train-centric reuse)
        self.N_z = None
        if build_N_z:
            self.N_z = [set() for _ in range(self.n_train)]
            for z_idx_prime in range(self.n_train):
                for node_id in self.L_train[z_idx_prime]:
                    member_idx = self.node_to_idx.get(int(node_id))
                    if member_idx is not None:
                        self.N_z[member_idx].add(z_idx_prime)
            print(f"  N_z inverse map built.")

        sizes = [len(L) for L in self.L_test]
        train_sizes = [len(L) for L in self.L_train]
        print(f"  Test-Centric Local sizes: min={min(sizes)}, "
              f"max={max(sizes)}, mean={np.mean(sizes):.1f}")
        print(f"  Train-Centric Local sizes: min={min(train_sizes)}, "
              f"max={max(train_sizes)}, mean={np.mean(train_sizes):.1f}")
        print(f"  max_L={self.max_L}")
        print(f"  Precomputation time: {time.time() - t0:.2f}s")
