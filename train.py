"""Trainingsloops für MLP und Random Forest."""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score

from models import MLP, RFModel
from config import MAX_EPOCHS, EARLY_STOPPING_PATIENCE


def _make_loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_mlp(config: dict, data: dict) -> dict:
    """Trainiert ein MLP und gibt AUROC, Accuracy und Trainingszeit zurück."""
    t0 = time.time()
    batch_size = config["batch_size"]
    train_loader = _make_loader(data["X_train"], data["y_train"], batch_size, shuffle=True)
    val_loader   = _make_loader(data["X_val"],   data["y_val"],   batch_size, shuffle=False)

    model = MLP(
        input_dim=data["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        activation=config["activation"],
        num_classes=data["num_classes"],
    )

    # Optimizer wählen
    optim_cls = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
    optimizer = optim_cls[config["optimizer_name"]](
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    best_val_auroc = 0.0
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        # Validation: Wahrscheinlichkeit für Klasse 1 berechnen
        model.eval()
        probs = []
        preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                logits = model(X_batch)
                prob = torch.softmax(logits, dim=1)[:, 1]
                probs.append(prob.numpy())
                preds.append(logits.argmax(dim=1).numpy())
        val_auroc    = roc_auc_score(data["y_val"], np.concatenate(probs))
        val_accuracy = (np.concatenate(preds) == data["y_val"]).mean()

        # Early Stopping
        if val_auroc > best_val_auroc:
            best_val_auroc    = val_auroc
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    return {
        "val_auroc":    float(best_val_auroc),
        "val_accuracy": float(best_val_accuracy),
        "train_time":   time.time() - t0,
    }


def train_rf(config: dict, data: dict) -> dict:
    """Trainiert einen Random Forest und gibt AUROC, Accuracy und Trainingszeit zurück."""
    t0 = time.time()
    model = RFModel(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        max_features=config["max_features"],
        min_samples_leaf=config["min_samples_leaf"],
        criterion=config["criterion"],
        max_samples=config["max_samples"],
    )
    model.model.fit(data["X_train"], data["y_train"])
    probs = model.model.predict_proba(data["X_val"])[:, 1]
    preds = model.model.predict(data["X_val"])
    return {
        "val_auroc":    float(roc_auc_score(data["y_val"], probs)),
        "val_accuracy": float((preds == data["y_val"]).mean()),
        "train_time":   time.time() - t0,
    }


def eval_test_mlp(config: dict, data: dict) -> float:
    """Trainiert MLP mit gegebener Konfiguration und gibt Test-AUROC zurück.

    Early Stopping basiert auf dem Validierungsset; das beste Modell wird
    am Ende auf dem Testset ausgewertet.
    """
    batch_size = config["batch_size"]
    train_loader = _make_loader(data["X_train"], data["y_train"], batch_size, shuffle=True)
    val_loader   = _make_loader(data["X_val"],   data["y_val"],   batch_size, shuffle=False)
    test_loader  = _make_loader(data["X_test"],  data["y_test"],  batch_size, shuffle=False)

    model = MLP(
        input_dim=data["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        activation=config["activation"],
        num_classes=data["num_classes"],
    )

    optim_cls = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
    optimizer = optim_cls[config["optimizer_name"]](
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    best_val_auroc = 0.0
    best_state     = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        probs = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                prob = torch.softmax(model(X_batch), dim=1)[:, 1]
                probs.append(prob.numpy())
        val_auroc = roc_auc_score(data["y_val"], np.concatenate(probs))

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    # bestes Modell laden und auf Testset auswerten
    model.load_state_dict(best_state)
    model.eval()
    probs = []
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            logits = model(X_batch)
            probs.append(torch.softmax(logits, dim=1)[:, 1].numpy())
            preds.append(logits.argmax(dim=1).numpy())
    return {
        "test_auroc":    float(roc_auc_score(data["y_test"], np.concatenate(probs))),
        "test_accuracy": float((np.concatenate(preds) == data["y_test"]).mean()),
    }


def eval_test_rf(config: dict, data: dict) -> dict:
    """Trainiert RF mit gegebener Konfiguration und gibt Test-AUROC und Test-Accuracy zurück."""
    model = RFModel(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        max_features=config["max_features"],
        min_samples_leaf=config["min_samples_leaf"],
        criterion=config["criterion"],
        max_samples=config["max_samples"],
    )
    model.model.fit(data["X_train"], data["y_train"])
    probs = model.model.predict_proba(data["X_test"])[:, 1]
    preds = model.model.predict(data["X_test"])
    return {
        "test_auroc":    float(roc_auc_score(data["y_test"], probs)),
        "test_accuracy": float((preds == data["y_test"]).mean()),
    }
