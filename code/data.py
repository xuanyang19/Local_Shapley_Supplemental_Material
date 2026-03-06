
import numpy as np
import random
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml


###############################################################################
# 1. Set random seed for reproducibility
###############################################################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

###############################################################################
# Helper: subsample with a dedicated RNG (so sampling is reproducible)
###############################################################################
def _subsample(X, y, n, rng):
    if n is None or len(X) <= n:
        return X, y
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], y[idx]

###############################################################################
# Helper: standardize train/test (fit on train only)
###############################################################################
def _standardize_train_test(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test

###############################################################################
# 2. Load Dataset Function
###############################################################################
def load_data(dataset, sample_size=None, test_size=0.3, random_state=42):
    """
    dataset: one of [
      "Breast_Cancer", "Iris", "MNIST", "MNIST_Pixels"
    ].

    sample_size: None, an int, or [train_size, test_size].
      - If int: use same size for train and test.
      - If [n_train, n_test]: subsample after splitting / loading.

    test_size: fraction for train_test_split if using tabular datasets.
    random_state: controls splitting and subsampling RNG.
    """
    if isinstance(sample_size, int):
        sample_size = [sample_size, sample_size]
    elif sample_size is None:
        sample_size = [None, None]
    elif isinstance(sample_size, list) and len(sample_size) == 2:
        pass
    else:
        raise ValueError("sample_size must be an int or a list of length 2.")

    rng = np.random.default_rng(random_state)

    # -----------------------
    # Existing datasets
    # -----------------------
    if dataset == "Breast_Cancer":
        data = load_breast_cancer()
        X = data.data.astype(np.float32)
        y = data.target.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_test = _standardize_train_test(X_train, X_test)

    elif dataset == "Iris":
        data = load_iris()
        X = data.data.astype(np.float32)
        y = data.target.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_test = _standardize_train_test(X_train, X_test)

    elif dataset == "MNIST":
        data = np.load("mnist_features.npz")
        X_train, y_train, X_test, y_test = (
            data["X_train"], data["y_train"],
            data["X_test"],  data["y_test"]
        )
        X_train = X_train.astype(np.float32)
        X_test  = X_test.astype(np.float32)
        y_train = y_train.astype(int)
        y_test  = y_test.astype(int)

    elif dataset == "MNIST_Pixels":
        """
        Raw MNIST pixels (28x28 flattened to 784).
        Loaded via OpenML, cached locally by sklearn.
        """
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml(
            name="mnist_784",
            version=1,
            as_frame=False,
            return_X_y=True
        )

        X = X.astype(np.float32)
        y = y.astype(int)

        # Normalize pixels to [0,1]
        X /= 255.0

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    else:
        raise ValueError(
            "Unsupported dataset. Choose from "
            "'Breast_Cancer','Iris','MNIST','MNIST_Pixels'."
        )

    # -----------------------
    # Optional sub-sampling (done LAST; reproducible)
    # -----------------------
    X_train, y_train = _subsample(X_train, y_train, sample_size[0], rng)
    X_test,  y_test  = _subsample(X_test,  y_test,  sample_size[1], rng)

    return X_train, y_train, X_test, y_test
