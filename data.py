"""Datensätze laden, splitten und skalieren."""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import TEST_SIZE, VAL_SIZE

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")


def _cache_path(name: str) -> str:
    return os.path.join(_CACHE_DIR, f"{name}.npz")


def _load_cache(name: str):
    path = _cache_path(name)
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return data["X"], data["y"]
    return None, None


def _save_cache(name: str, X: np.ndarray, y: np.ndarray) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.savez_compressed(_cache_path(name), X=X, y=y)


def load_data(seed: int, dataset: str = "breast_cancer") -> dict:
    """Lädt einen Datensatz und gibt ihn als 60/20/20-Split zurück."""

    if dataset == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
        X = X.astype(np.float32)

    elif dataset == "diabetes":
        X, y = _load_cache("diabetes")
        if X is None:
            X, y = fetch_openml(name="diabetes", version=1, return_X_y=True, as_frame=False)
            X = X.astype(np.float32)
            y = (y == "tested_positive").astype(int)
            _save_cache("diabetes", X, y)

    elif dataset == "bank_marketing":
        X, y = _load_cache("bank_marketing")
        if X is None:
            X, y = fetch_openml(name="bank-marketing", version=1, as_frame=True, return_X_y=True)
            X = pd.get_dummies(X).fillna(0).values.astype(np.float32)
            y = (y.astype(int) == 2).astype(int).values  # 2="yes" → 1
            _save_cache("bank_marketing", X, y)

        # auf 10% subsamplen
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=len(X) // 10, replace=False)
        X, y = X[idx], y[idx]

    elif dataset == "german_credit":
        X, y = _load_cache("german_credit")
        if X is None:
            X, y = fetch_openml(name="credit-g", version=1, as_frame=True, return_X_y=True)
            X = pd.get_dummies(X).fillna(0).values.astype(np.float32)
            y = (y == "good").astype(int).values  # "good" → 1
            _save_cache("german_credit", X, y)

    elif dataset == "adult_income":
        X, y = _load_cache("adult_income")
        if X is None:
            X, y = fetch_openml(name="adult", version=2, as_frame=True, return_X_y=True)
            X = pd.get_dummies(X).fillna(0).values.astype(np.float32)
            y = (y.astype(str) == ">50K").astype(int).values  # ">50K" → 1
            _save_cache("adult_income", X, y)

        # auf 10% subsamplen
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=len(X) // 10, replace=False)
        X, y = X[idx], y[idx]

    # 60/20/20 Split (stratifiziert)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=seed, stratify=y_train
    )

    # Standardisierung: fit nur auf Trainingsdaten
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "input_dim":   X_train.shape[1],
        "num_classes": len(np.unique(y)),
    }
