import numpy as np
import pandas as pd
from utils import one_hot, split_data


def normalize(X, method="zscore"):
    """Normalize features using z-score or min-max scaling."""
    if method == "zscore":
        return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    elif method == "minmax":
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def softmax(x):
    """Numerically stable softmax for internal use."""
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def generate_classification_data(n_samples=1000, n_features=2, n_classes=2, seed=42, normalize_data=True):
    """
    Generates synthetic classification data using Gaussian blobs.
    """
    assert n_classes >= 2, "n_classes must be >= 2"
    np.random.seed(seed)

    X = np.random.randn(n_samples, n_features)
    W = np.random.randn(n_features, n_classes)
    b = np.random.randn(1, n_classes)

    logits = np.dot(X, W) + b
    probs = softmax(logits)
    y = np.argmax(probs, axis=1)

    if normalize_data:
        X = normalize(X)

    return X, y

def clean_data(df):
    """
    Drops rows with any NaN or Inf values.
    """
    df = df.replace([np.inf, -np.inf], np.nan)  # replace infs first
    df_cleaned = df.dropna()  # drop rows with NaNs
    return df_cleaned

def load_csv_data(data_path, label_column, test_ratio=0.2, one_hot_labels=False, normalize_data=True):
    """
    Loads data from CSV. Assumes labels are in `label_column`.
    """
    df = pd.read_csv(data_path)
    df = clean_data(df) 
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in CSV.")

    X = df.drop(columns=[label_column]).values
    y = df[label_column].values

    if normalize_data:
        X = normalize(X)

    if one_hot_labels:
        y = one_hot(y.astype(int))

    # Reshape y if it's binary and not one-hot
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return split_data(X, y, test_ratio=test_ratio)




def load_data(data_path, test_ratio=0.2, one_hot_labels=False):
    df = pd.read_csv(data_path)

    df = clean_data(df) 
    
    # Assuming last column is label
    X = df.drop(columns=["TenYearCHD"]).values
    y = df["TenYearCHD"].values.reshape(-1, 1)

    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    if one_hot_labels:
        y = one_hot(y.astype(int))

    X_train, X_test, y_train, y_test = split_data(X, y, test_ratio=test_ratio)
    return X_train, X_test, y_train, y_test
