"""Datensätze laden, splitten und skalieren."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import TEST_SIZE, VAL_SIZE


def load_data(seed: int, dataset: str = "breast_cancer") -> dict:
    """Lädt einen Datensatz und gibt ihn als 60/20/20-Split zurück."""

    if dataset == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
        X = X.astype(np.float32)

    elif dataset == "diabetes":
        X, y = fetch_openml(name="diabetes", version=1, return_X_y=True, as_frame=False)
        X = X.astype(np.float32)
        y = (y == "tested_positive").astype(int)  # "tested_positive" → 1

    elif dataset == "bank_marketing":
        X, y = fetch_openml(name="bank-marketing", version=1, as_frame=True, return_X_y=True)
        X = pd.get_dummies(X).fillna(0).values.astype(np.float32)
        y = (y.astype(int) == 2).astype(int).values  # 2="yes" → 1

        # auf 10% subsamplen
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=len(X) // 10, replace=False)
        X, y = X[idx], y[idx]

    elif dataset == "german_credit":
        X, y = fetch_openml(name="credit-g", version=1, as_frame=True, return_X_y=True)
        X = pd.get_dummies(X).fillna(0).values.astype(np.float32)
        y = (y == "good").astype(int).values  # "good" → 1

    elif dataset == "adult_income":
        X, y = fetch_openml(name="adult", version=2, as_frame=True, return_X_y=True)
        X = pd.get_dummies(X).fillna(0).values.astype(np.float32)
        y = (y.astype(str) == ">50K").astype(int).values  # ">50K" → 1

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
