import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

# ----------------------------
# Helpers you likely already have
# ----------------------------

def _top_r_neighbors(X_train: np.ndarray, x: np.ndarray, r: int) -> np.ndarray:
    """Return indices of r nearest neighbors in X_train to x (Euclidean)."""
    r = max(1, min(r, X_train.shape[0]))
    d2 = np.sum((X_train - x[None, :]) ** 2, axis=1)
    return np.argpartition(d2, r - 1)[:r].astype(int)

def _top_r_neighbors_restricted(X_train: np.ndarray, x: np.ndarray, candidates: np.ndarray, r: int) -> np.ndarray:
    """Return indices of r nearest neighbors to x, restricted to 'candidates' indices."""
    candidates = np.asarray(candidates, dtype=int)
    if candidates.size == 0:
        return np.array([], dtype=int)
    r = max(1, min(r, candidates.size))
    Xc = X_train[candidates]
    d2 = np.sum((Xc - x[None, :]) ** 2, axis=1)
    local = np.argpartition(d2, r - 1)[:r]
    return candidates[local].astype(int)

def _leaf_memberships_single_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    tree_kwargs: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a single decision tree and return leaf assignments.
    Returns:
      - train_leaf: shape (n_train,), leaf id for each train sample
      - test_leaf:  shape (n_test,),  leaf id for each test sample
    """
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(random_state=seed, **tree_kwargs)
    clf.fit(X_train, y_train)
    train_leaf = clf.apply(X_train).astype(int)
    test_leaf = clf.apply(X_test).astype(int)
    return train_leaf, test_leaf

def _build_leaf_to_train_indices(train_leaf: np.ndarray) -> Dict[int, np.ndarray]:
    """Map leaf_id -> np.array(train_indices)."""
    leaf_to_idx: Dict[int, List[int]] = {}
    for i, lid in enumerate(train_leaf.tolist()):
        leaf_to_idx.setdefault(lid, []).append(i)
    return {lid: np.asarray(idxs, dtype=int) for lid, idxs in leaf_to_idx.items()}


# ----------------------------
# Core implementation
# ----------------------------

@dataclass(frozen=True)
class LeafFreqConfig:
    """
    Config for leaf_mode="freq" behavior.
    n_trees: number of independently built trees (seed + j)
    splitter_random: if True, force 'splitter="random"' when not specified, encouraging diversity
    """
    n_trees: int = 10
    splitter_random: bool = True


def precompute_local_neighbors_strict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    neighbor_method: str = "leaf",         # "leaf" or "knn" (legacy)
    leaf_mode: str = "freq",             # "single" or "freq"
    freq_cfg: LeafFreqConfig = LeafFreqConfig(),
    desired_min: int = 10,
    desired_max: int = 20,
    knn_k: int = 15,
    seed: int = 42,
    tree_kwargs: Optional[dict] = None,
    # In freq mode, decide final selection:
    # - "freq": take top desired_max by frequency (ties by smaller index)
    # - "freq_then_knn": build a frequency-ranked candidate pool, then take KNN within that pool
    freq_select: str = "freq_then_knn",
    freq_pool_cap: Optional[int] = None,   # if freq_then_knn: optionally cap candidate pool size
    save_path: Optional[str] = None,
    load_path: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Compute local neighbors (train indices) for each test point.

    Returns:
      L_t: list length n_test, each entry is np.array of train indices for test point t.

    Definitions (matches your spec exactly):
    - Leaf neighbors: local region = set of train samples that fall in the *same leaf node*
      as the test sample in a decision tree.
    - leaf_mode:
        * "single": build one tree once, use its leaf co-membership.
        * "freq": build multiple trees, count how often each train sample co-appears with
          the test sample in the same leaf; rank by frequency descending (tie by index ascending).
    - Size stabilization target range [desired_min, desired_max] (e.g., 5-10):
        * If leaf_size in range: return leaf set (or top by freq in freq mode if > desired_max).
        * If leaf_size < desired_min:
              do KNN over ALL train, and INCLUDE the leaf nodes.
              Then enforce stability by truncating to desired_max (using distance among merged).
        * If leaf_size > desired_max:
              do KNN WITHIN the leaf nodes (or within freq-ranked pool), returning desired_max.
    - neighbor_method="knn" (legacy):
        return KNN of size desired_max over all train.

    Notes:
    - In leaf_mode="freq", you have two reasonable final selectors:
        * freq_select="freq": neighbors = top desired_max by frequency (most literal).
        * freq_select="freq_then_knn": treat frequency-ranked set as candidate pool, then KNN
          within it for desired_max (often better when feature-space distance matters).
    """
    if desired_min <= 0 or desired_max <= 0 or desired_min > desired_max:
        raise ValueError("Require 0 < desired_min <= desired_max.")
    if freq_select not in ("freq", "freq_then_knn"):
        raise ValueError(f"freq_select must be 'freq' or 'freq_then_knn', got {freq_select}")
    if leaf_mode not in ("single", "freq"):
        raise ValueError(f"leaf_mode must be 'single' or 'freq', got {leaf_mode}")

    # Load cache
    if load_path is not None and os.path.exists(load_path):
        obj = np.load(load_path, allow_pickle=True)
        L_t = obj["L_t"].tolist()
        return [np.asarray(a, dtype=int) for a in L_t]

    tree_kwargs = tree_kwargs or {}
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Legacy KNN-only behavior
    if neighbor_method.lower() == "knn":
        out: List[np.ndarray] = []
        for t in range(n_test):
            out.append(_top_r_neighbors(X_train, X_test[t], desired_max))
        _maybe_save_neighbors(out, save_path)
        return out

    # --- Build raw candidate sets ---
    if leaf_mode == "single":
        train_leaf, test_leaf = _leaf_memberships_single_tree(
            X_train, y_train, X_test, seed=seed, tree_kwargs=tree_kwargs
        )
        leaf_to_idx = _build_leaf_to_train_indices(train_leaf)
        raw_sets = [leaf_to_idx.get(int(test_leaf[t]), np.array([], dtype=int)) for t in range(n_test)]

    else:
        # frequency across multiple trees
        counts_per_test = [np.zeros(n_train, dtype=np.int32) for _ in range(n_test)]
        tree_kwargs_freq = dict(tree_kwargs)
        if freq_cfg.splitter_random:
            tree_kwargs_freq.setdefault("splitter", "random")

        for j in range(max(1, int(freq_cfg.n_trees))):
            train_leaf, test_leaf = _leaf_memberships_single_tree(
                X_train, y_train, X_test, seed=seed + j, tree_kwargs=tree_kwargs_freq
            )
            leaf_to_idx = _build_leaf_to_train_indices(train_leaf)

            for t in range(n_test):
                members = leaf_to_idx.get(int(test_leaf[t]), None)
                if members is not None and members.size > 0:
                    counts_per_test[t][members] += 1
        raw_sets = []
        for t in range(n_test):
            c = counts_per_test[t]
            nz = np.flatnonzero(c)
            if nz.size == 0:
                raw_sets.append(np.array([], dtype=int))
                continue
            # frequency desc, tie by index asc
            order = np.lexsort((nz, -c[nz]))
            ranked = nz[order].astype(int)

            if freq_pool_cap is not None:
                ranked = ranked[: int(freq_pool_cap)]
            raw_sets.append(ranked)

    # --- Stabilize neighborhood size strictly to [desired_min, desired_max] ---
    L_t: List[np.ndarray] = [np.array([], dtype=int) for _ in range(n_test)]

    for t in range(n_test):
        cand = np.asarray(raw_sets[t], dtype=int)
        m = cand.size

        # CASE 1: too small -> global KNN + include leaf (union), then truncate to desired_max by distance
        if m < desired_min:
            knn_all = _top_r_neighbors(X_train, X_test[t], knn_k)
            merged = np.unique(np.concatenate([cand, knn_all])) if m > 0 else knn_all
            # enforce stable upper bound desired_max (distance-based selection within merged)
            if merged.size > desired_max:
                merged = _top_r_neighbors_restricted(X_train, X_test[t], merged, desired_max)
            L_t[t] = merged.astype(int)

        # CASE 2: in range -> accept as is (but in freq mode you may want strict size)
        elif desired_min <= m <= desired_max:
            # In freq mode, cand is already ranked by freq; returning all is OK since in range.
            L_t[t] = cand.astype(int)

        # CASE 3: too large -> KNN within candidates (leaf or freq pool), size desired_max
        else:
            if leaf_mode == "freq" and freq_select == "freq":
                # Most literal: use frequency ranking directly (top desired_max)
                L_t[t] = cand[:desired_max].astype(int)
            else:
                # Use distance-based KNN within candidate pool
                L_t[t] = _top_r_neighbors_restricted(X_train, X_test[t], cand, desired_max).astype(int)

        # Final safety: ensure not exceeding desired_max; if somehow smaller than desired_min, keep what we have
        if L_t[t].size > desired_max:
            L_t[t] = _top_r_neighbors_restricted(X_train, X_test[t], L_t[t], desired_max)

    _maybe_save_neighbors(L_t, save_path)
    return L_t


def _maybe_save_neighbors(L_t: List[np.ndarray], save_path: Optional[str]) -> None:
    if save_path is None:
        return
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez_compressed(save_path, L_t=np.array(L_t, dtype=object))
