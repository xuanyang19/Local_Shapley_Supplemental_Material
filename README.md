# Local Shapley

Code for computing Shapley values with local structure, comparing five methods across four model families.

## Methods

| Method | Key | Description |
|--------|-----|-------------|
| Global MC | `global_mc` | Standard Monte Carlo Shapley via full permutations |
| TMC-Shapley | `tmc` | Truncated Monte Carlo with patience-based early stopping |
| Complementary | `complementary` | Paired v(S) - v(S^c) with stratified binning |
| Local MC | `local_mc` | Monte Carlo restricted to local neighborhoods |
| LSMR-A | `lsmr_a` | Local Shapley with importance-weighted reuse across test points |

## Models

| Module | Model | Default Dataset |
|--------|-------|-----------------|
| `wknn` | Weighted K-NN (K=5) | MNIST |
| `svm` | RBF-SVM | Breast_Cancer |
| `tree` | Decision Tree | Iris |
| `gnn` | GCN (2-layer) | Cora |
| `template` | User-defined | Iris |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run from the `Local_Shapley/` directory:

```bash
# WKNN on MNIST
python -m wknn --dataset MNIST --methods global_mc lsmr_a --seeds 42

# SVM on Breast Cancer
python -m svm --dataset Breast_Cancer --methods global_mc lsmr_a --seeds 42

# Decision Tree on Iris
python -m tree --dataset Iris --methods global_mc lsmr_a --seeds 42

# GNN on Cora
python -m gnn --methods global_mc lsmr_a --seeds 42

# Template with custom model
python -m template --model-module template.examples.svm_model --dataset Iris --methods global_mc lsmr_a
```

### Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name | varies by module |
| `--sample-size N_TR N_TE` | Train/test sample sizes | 1000 1000 |
| `--seeds S [S ...]` | Random seeds | 42 |
| `--methods M [M ...]` | Methods to run | all five |
| `--max-samples` | Maximum MC samples | varies by module |
| `--threshold` | Convergence threshold | 0.05 |
| `--out-dir` | Output directory | `{module}_results` |

## Output

Each method saves to `{out_dir}/{method_key}/`:
- `{tag}_phi_{dataset}_seed{seed}_n{samples}.npy` — Shapley value vector
- `{tag}_stats_{dataset}_seed{seed}_n{samples}.json` — convergence history and metadata
