"""Hyperparameter-Optimierung: Bayesian Search (TPE), Random Search, CMA-ES und ACRS."""

import numpy as np
import optuna

from config import (N_TRIALS_MLP, N_TRIALS_RF, ACRS_R, ACRS_L, ACRS_ALPHA,
                    MLP_SEARCH_SPACE, RF_SEARCH_SPACE)
from train import train_mlp, train_rf

# Optuna-Logs auf Warnungen reduzieren (Trial-Fortschritt nicht ausgeben)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _budget(model_type: str) -> int:
    """Gibt das Trial-Budget für das jeweilige Modell zurück."""
    return N_TRIALS_MLP if model_type == "mlp" else N_TRIALS_RF


def _suggest_mlp(trial) -> dict:
    """Schlägt eine MLP-Konfiguration via Optuna-Trial vor."""
    return {
        "learning_rate":  trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "batch_size_exp": trial.suggest_int("batch_size_exp", 4, 7),
        "hidden_dim":     trial.suggest_int("hidden_dim", 32, 256),
        "dropout":        trial.suggest_float("dropout", 0.0, 0.5),
        "num_layers":     trial.suggest_categorical("num_layers", [1, 2, 3]),
        "optimizer_name": trial.suggest_categorical("optimizer_name", ["adam", "sgd", "adamw"]),
        "weight_decay":   trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "activation":     trial.suggest_categorical("activation", ["relu", "tanh"]),
    }


def _suggest_rf(trial) -> dict:
    """Schlägt eine RF-Konfiguration via Optuna-Trial vor."""
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "criterion":         trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_samples":       trial.suggest_float("max_samples", 0.5, 1.0),
    }


def _run_optuna_study(data: dict, model_type: str, sampler, seed: int = None) -> tuple:
    """Führt eine HPO-Optimierung mit dem gegebenen Sampler durch und gibt Konfiguration, AUROC und Trial-History zurück."""
    suggest_fn = _suggest_mlp if model_type == "mlp" else _suggest_rf
    train_fn   = train_mlp    if model_type == "mlp" else train_rf

    # Black-Box-Objective-Funktion für Optuna: nimmt eine Konfiguration, trainiert das Modell und gibt die Val-AUROC zurück
    def objective(trial):
        # Optuna schlägt eine Konfiguration vor (abhängig vom Sampler)
        config = suggest_fn(trial)
        result = train_fn(config, data, seed) if model_type == "rf" else train_fn(config, data)
        # Zusätzliche Metriken am Trial speichern (Val-Accuracy, Trainingszeit) für spätere Auswertung
        trial.set_user_attr("val_accuracy", result["val_accuracy"])
        trial.set_user_attr("train_time",   result["train_time"])
        return result["val_auroc"]

    # Sampler bestimmt die Such-Strategie (TPE, Random, CMA-ES)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=_budget(model_type))

    # Trial-History für spätere Auswertung aufbauen
    trial_history = [
        {
            "trial_nr":     t.number + 1,
            "val_auroc":    t.value,
            "val_accuracy": t.user_attrs["val_accuracy"],
            "train_time":   t.user_attrs["train_time"],
            "config":       t.params,
        }
        for t in study.trials
    ]

    return study.best_params, study.best_value, trial_history


def bayesian_search(data: dict, model_type: str, seed: int) -> tuple:
    """Bayesian Optimization via TPE (Optuna).

    Gibt die beste gefundene Konfiguration, den besten AUROC und die Trial-History zurück.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    return _run_optuna_study(data, model_type, sampler, seed=seed)


def random_search(data: dict, model_type: str, seed: int) -> tuple:
    """Random Search via Optuna.

    Gibt die beste gefundene Konfiguration, den besten AUROC und die Trial-History zurück.
    """
    sampler = optuna.samplers.RandomSampler(seed=seed)
    return _run_optuna_study(data, model_type, sampler, seed=seed)


def cmaes_search(data: dict, model_type: str, seed: int) -> tuple:
    """CMA-ES via Optuna.

    Gibt die beste gefundene Konfiguration, den besten AUROC und die Trial-History zurück.
    """
    sampler = optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=False)
    return _run_optuna_study(data, model_type, sampler, seed=seed)


def _sample(hp: dict, rng, best_val=None) -> float | int | str:
    """Sampelt einen zufälligen Wert aus einem HP-Eintrag des ACRS-Suchraums.

    Wenn best_val übergeben wird, wird Normal-Sampling um den Bestwert verwendet.
    Kategoriale HPs werden immer uniform gesampelt.
    """
    if best_val is None or hp["type"] == "categorical":
        # uniform
        if hp["type"] == "float":
            return rng.uniform(hp["low"], hp["high"])
        elif hp["type"] == "log":
            return float(np.exp(rng.uniform(np.log(hp["low"]), np.log(hp["high"]))))
        elif hp["type"] == "int":
            return int(rng.integers(hp["low"], hp["high"] + 1))
        elif hp["type"] == "categorical":
            return hp["choices"][rng.integers(len(hp["choices"]))]
    else:
        # Normal-Sampling um best_val
        if hp["type"] == "float":
            std = (hp["high"] - hp["low"]) / 4
            return float(np.clip(rng.normal(best_val, std), hp["low"], hp["high"]))
        elif hp["type"] == "log":
            std = (np.log(hp["high"]) - np.log(hp["low"])) / 4
            return float(np.exp(np.clip(rng.normal(np.log(best_val), std), np.log(hp["low"]), np.log(hp["high"]))))
        elif hp["type"] == "int":
            std = (hp["high"] - hp["low"]) / 4
            return int(np.clip(np.round(rng.normal(best_val, std)), hp["low"], hp["high"]))


def _shrink(hp: dict, best_val, alpha: float) -> dict:
    """Verkleinert den Suchbereich eines HP um den aktuellen Bestwert."""
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
    keys = rng.random(n) / (weights + 1e-10)
    return np.argsort(keys)


def acrs_search(data: dict, model_type: str, seed: int, sampling: str = "uniform", shrink: bool = False) -> tuple:
    """Adaptive Coordinate Random Search (ACRS).

    sampling: "uniform" (standard) oder "normal" (Gauß um best_val).
    shrink: ob der Suchbereich verkleinert werden soll.
    Gibt die beste Konfiguration, den besten AUROC und die Trial-History zurück.
    """
    search_space = MLP_SEARCH_SPACE if model_type == "mlp" else RF_SEARCH_SPACE
    train_fn = train_mlp if model_type == "mlp" else train_rf
    rng = np.random.default_rng(seed)
    hp_names = list(search_space.keys())
    n = len(hp_names)

    trial_history = []
    trial_nr      = 0

    def _log_trial(config: dict, result: dict):
        """Hängt einen Trial-Eintrag an die History an."""
        trial_history.append({
            "trial_nr":     trial_nr,
            "val_auroc":    result["val_auroc"],
            "val_accuracy": result["val_accuracy"],
            "train_time":   result["train_time"],
            "config":       config,
        })

    # Initialisierung: zufällige Startkonfiguration
    best_config    = {name: _sample(search_space[name], rng) for name in hp_names}
    result         = train_fn(best_config, data)
    trial_nr      += 1
    best_auroc     = result["val_auroc"]
    weights        = np.full(n, 1 / n)
    current_ranges = {name: search_space[name].copy() for name in hp_names}
    _log_trial(best_config, result)

    # Hauptschleife: R Runden über alle HPs
    for _ in range(ACRS_R):
        deltas = np.zeros(n)
        permutation = _priority_order(n, weights, rng)

        # koordinatenweise Optimierung in gewichteter Reihenfolge
        for j in permutation:
            hp_name = hp_names[j]
            y_before = best_auroc

            # L Kandidaten für diesen HP samplen und evaluieren
            for _ in range(ACRS_L):
                candidate = best_config.copy()
                bv = best_config[hp_name] if sampling == "normal" else None
                candidate[hp_name] = _sample(current_ranges[hp_name], rng, best_val=bv)
                result = train_fn(candidate, data, seed) if model_type == "rf" else train_fn(candidate, data)
                y = result["val_auroc"]
                trial_nr += 1
                _log_trial(candidate, result)

                if y > best_auroc:
                    best_auroc = y
                    best_config = candidate

            deltas[j] = best_auroc - y_before
            
            if shrink:
                current_ranges[hp_name] = _shrink(current_ranges[hp_name], best_config[hp_name], ACRS_ALPHA)

        # Gewichte für nächste Runde aktualisieren
        total = deltas.sum()
        weights = deltas / total if total > 0 else np.full(n, 1 / n)

    return best_config, best_auroc, trial_history
