#!/usr/bin/env python3
"""
GCN model definition and train/eval utility.
- GCN: 2-layer GCNConv with ReLU + log-softmax
- train_and_eval_gnn: trains a fresh GCN on a subset and returns
  per-test-node binary accuracy vector
- extract_subgraph, get_k_hop_neighbors
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph, subgraph


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def get_k_hop_neighbors(node_idx, k, edge_index):
    subset, _, _, _ = k_hop_subgraph(node_idx, k, edge_index, relabel_nodes=False)
    return subset


def extract_subgraph(subset_nodes, edge_index, num_nodes, test_indices, device):
    all_test_nodes = torch.tensor(test_indices, dtype=torch.long, device=device)
    if len(subset_nodes) > 0:
        subset_tensor = torch.tensor(subset_nodes, dtype=torch.long, device=device)
        combined_subset = torch.cat([subset_tensor, all_test_nodes]).unique()
    else:
        combined_subset = all_test_nodes.unique()

    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_index_sub, _ = subgraph(combined_subset, edge_index_t, relabel_nodes=True)

    subgraph_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    subgraph_mask[combined_subset] = True

    return edge_index_sub, subgraph_mask, combined_subset


def train_and_eval_gnn(subset_nodes, x_np, y_np, edge_index_np, test_indices,
                       n_classes, seed, epochs, device):
    """Train a fresh GCN on subset_nodes and return binary accuracy per test node."""
    n_test = len(test_indices)
    num_nodes = len(y_np)

    if len(subset_nodes) == 0:
        return np.full(n_test, 1.0 / n_classes, dtype=float)

    edge_index_sub, subgraph_mask, combined_subset = extract_subgraph(
        subset_nodes, edge_index_np, num_nodes, test_indices, device
    )

    x = torch.tensor(x_np, dtype=torch.float, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)

    subset_train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    subset_train_mask[subset_nodes] = True
    subset_train_mask = subset_train_mask & subgraph_mask

    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask[test_indices] = True

    torch.manual_seed(seed)
    model = GCN(x.shape[1], 32, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(x[subgraph_mask], edge_index_sub)
        subgraph_train_mask = subset_train_mask[subgraph_mask]
        if subgraph_train_mask.sum() > 0:
            subgraph_y = y[subgraph_mask]
            loss = F.nll_loss(out[subgraph_train_mask], subgraph_y[subgraph_train_mask])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(x[subgraph_mask], edge_index_sub)
        subgraph_test_mask = test_mask[subgraph_mask]
        pred = out[subgraph_test_mask].argmax(dim=1)
        subgraph_y = y[subgraph_mask]
        correct_vec = (pred == subgraph_y[subgraph_test_mask]).float().cpu().numpy()

    # Force clear PyTorch tensors to prevent worker GPU OOM
    if 'loss' in locals():
        del loss
    del model, optimizer, out
    del x, y, edge_index_sub, subgraph_mask, combined_subset
    del subset_train_mask, test_mask, subgraph_test_mask, pred, subgraph_y

    return correct_vec
