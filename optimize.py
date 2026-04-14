"""Hyperparameter-Optimierung: Bayesian Search (TPE) und ACRS."""

import os
import numpy as np
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch

from config import (N_TRIALS, ACRS_R, ACRS_L, ACRS_ALPHA,
                    MLP_SEARCH_SPACE, RF_SEARCH_SPACE,
                    MLP_SEARCH_SPACE_ACRS, RF_SEARCH_SPACE_ACRS)
from train import train_mlp, train_rf

# Projektverzeichnis für Ray-Worker sichtbar machen
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _sample(hp: dict, rng) -> float | int | str:
    """Sampelt einen zufälligen Wert aus einem HP-Eintrag des ACRS-Suchraums."""
    if hp["type"] == "float":
        return rng.uniform(hp["low"], hp["high"])
    elif hp["type"] == "log":
        return float(np.exp(rng.uniform(np.log(hp["low"]), np.log(hp["high"]))))
    elif hp["type"] == "int":
        return int(rng.integers(hp["low"], hp["high"] + 1))
    elif hp["type"] == "categorical":
        return hp["choices"][rng.integers(len(hp["choices"]))]


def _shrink(hp: dict, best_val, alpha: float) -> dict:
    """Kontrahiert den Suchbereich eines HP um den aktuellen Bestwert."""
    if hp["type"] == "categorical":
        return hp

    new_hp = hp.copy()
    if hp["type"] in ("float", "int"):
        a, b = hp["low"], hp["high"]
        new_low  = best_val - alpha * (best_val - a)
        new_high = best_val + alpha * (b - best_val)
        if hp["type"] == "int":
            new_low  = int(np.floor(new_low))
            new_high = int(np.ceil(new_high))
            if new_high <= new_low:
                new_high = new_low + 1
    elif hp["type"] == "log":
        log_a, log_b = np.log(hp["low"]), np.log(hp["high"])
        log_best     = np.log(best_val)
        new_low  = float(np.exp(log_best - alpha * (log_best - log_a)))
        new_high = float(np.exp(log_best + alpha * (log_b - log_best)))

    new_hp["low"]  = max(hp["low"],  new_low)
    new_hp["high"] = min(hp["high"], new_high)
    return new_hp


def _priority_order(n: int, weights: np.ndarray, rng) -> np.ndarray:
    """Gibt eine gewichtete zufällige Reihenfolge der HP-Indizes zurück."""
    keys = rng.random(n) / (weights + 1e-10) # Teilen durch 0 vermeiden
    return np.argsort(keys)


def bayesian_search(data: dict, model_type: str, seed: int) -> dict:
    """Bayesian Optimization via TPE (Optuna) mit Ray Tune.
    
    Gibt die beste gefundene Konfiguration und den besten AUROC zurück.
    """
    ray.init(runtime_env={"working_dir": _PROJECT_DIR}, ignore_reinit_error=True)

    search_space = MLP_SEARCH_SPACE if model_type == "mlp" else RF_SEARCH_SPACE
    train_fn = train_mlp if model_type == "mlp" else train_rf

    def trainable(config):
        val_auroc = train_fn(config, data)
        tune.report({"val_auroc": val_auroc})

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(metric="val_auroc", mode="max", seed=seed),
            num_samples=N_TRIALS,
            metric="val_auroc",
            mode="max",
            max_concurrent_trials=1,  # sequentiell: jeder Trial konditioniert auf alle vorherigen
        ),
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="val_auroc", mode="max")
    return best_result.config, best_result.metrics["val_auroc"]


def acrs_search(data: dict, model_type: str, seed: int) -> tuple:
    """Adaptive Coordinate Random Search (ACRS).

    Gibt die beste gefundene Konfiguration und den besten AUROC zurück.
    """
    search_space = MLP_SEARCH_SPACE_ACRS if model_type == "mlp" else RF_SEARCH_SPACE_ACRS
    train_fn = train_mlp if model_type == "mlp" else train_rf
    rng = np.random.default_rng(seed)
    hp_names = list(search_space.keys())
    n = len(hp_names)

    # Initialisierung: zufällige Startkonfiguration
    best_config    = {name: _sample(search_space[name], rng) for name in hp_names}
    best_auroc     = train_fn(best_config, data)
    weights        = np.full(n, 1 / n)
    current_ranges = {name: search_space[name].copy() for name in hp_names}

    for r in range(ACRS_R):
        deltas = np.zeros(n)
        permutation  = _priority_order(n, weights, rng)

        for j in permutation:
            hp_name  = hp_names[j]
            y_before = best_auroc

            for l in range(ACRS_L):
                candidate = best_config.copy()
                candidate[hp_name] = _sample(current_ranges[hp_name], rng)
                y = train_fn(candidate, data)
                if y > best_auroc:
                    best_auroc  = y
                    best_config = candidate

            deltas[j] = best_auroc - y_before
            current_ranges[hp_name] = _shrink(current_ranges[hp_name], best_config[hp_name], ACRS_ALPHA)

        total = deltas.sum()
        weights = deltas / total if total > 0 else np.full(n, 1 / n)

    return best_config, best_auroc
