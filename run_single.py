"""Führt einen einzelnen Run aus und speichert Ergebnisse in results/partial/."""

import argparse
import csv
import json
import os
import time

from codecarbon import EmissionsTracker

from data import load_data
from optimize import bayesian_search, acrs_search, random_search, cmaes_search
from train import eval_test_mlp, eval_test_rf

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PARTIAL_DIR = os.path.join(RESULTS_DIR, "partial")

RUNS_FIELDNAMES = ["dataset", "seed", "method", "model_type",
                   "val_auroc", "val_accuracy", "test_auroc", "test_accuracy",
                   "duration_s", "energy_kwh", "best_config"]

TRIALS_FIELDNAMES = ["trial_nr", "dataset", "model", "optimizer", "seed",
                     "val_auroc", "val_accuracy", "train_time", "config"]


def _run(method: str, model_type: str, data: dict, seed: int) -> dict:
    """Führt einen einzelnen Run durch und gibt die Ergebnisse als Dict zurück."""
    tracker = EmissionsTracker(save_to_file=False, logging_logger=None, log_level="error")
    tracker.start()
    t0 = time.time()

    if method == "bo":
        best_config, val_auroc, trial_history = bayesian_search(data, model_type=model_type, seed=seed)
    elif method == "rs":
        best_config, val_auroc, trial_history = random_search(data, model_type=model_type, seed=seed)
    elif method == "cmaes":
        best_config, val_auroc, trial_history = cmaes_search(data, model_type=model_type, seed=seed)
    else:
        best_config, val_auroc, trial_history = acrs_search(data, model_type=model_type, seed=seed)

    duration   = time.time() - t0
    tracker.stop()
    energy_kwh = tracker._total_energy.kWh

    if model_type == "mlp":
        test_result = eval_test_mlp(best_config, data)
    else:
        test_result = eval_test_rf(best_config, data)
    test_auroc    = test_result["test_auroc"]
    test_accuracy = test_result["test_accuracy"]

    best_trial   = max(trial_history, key=lambda t: t["val_auroc"])
    val_accuracy = best_trial["val_accuracy"]

    return {
        "val_auroc":     round(float(val_auroc),     4),
        "val_accuracy":  round(float(val_accuracy),  4),
        "test_auroc":    round(float(test_auroc),    4),
        "test_accuracy": round(float(test_accuracy), 4),
        "duration_s":    round(duration,             1),
        "energy_kwh":    round(energy_kwh,           6),
        "best_config":   json.dumps(best_config),
        "trial_history": trial_history,
    }


def run_single(dataset: str, model_type: str, method: str, seed: int):
    """Führt einen einzelnen Run durch und speichert in results/partial/."""
    os.makedirs(PARTIAL_DIR, exist_ok=True)

    prefix      = f"{dataset}_{model_type}_{method}_{seed}"
    runs_file   = os.path.join(PARTIAL_DIR, f"runs_{prefix}.csv")
    trials_file = os.path.join(PARTIAL_DIR, f"trials_{prefix}.csv")

    if os.path.exists(runs_file) and os.path.getsize(runs_file) > 0:
        print(f"  Übersprungen (bereits vorhanden): {prefix}")
        return

    data   = load_data(seed=seed, dataset=dataset)
    print(f"  {method.upper()} {model_type.upper()} | {dataset} | seed={seed} ...", end=" ", flush=True)

    result        = _run(method, model_type, data, seed)
    trial_history = result.pop("trial_history")

    print(f"val={result['val_auroc']:.4f}  test={result['test_auroc']:.4f}  {result['duration_s']:.0f}s")

    with open(runs_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_FIELDNAMES)
        writer.writeheader()
        writer.writerow({"dataset": dataset, "seed": seed, "method": method,
                         "model_type": model_type, **result})

    with open(trials_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIALS_FIELDNAMES)
        writer.writeheader()
        for trial in trial_history:
            writer.writerow({
                "trial_nr":     trial["trial_nr"],
                "dataset":      dataset,
                "model":        model_type,
                "optimizer":    method,
                "seed":         seed,
                "val_auroc":    trial["val_auroc"],
                "val_accuracy": trial["val_accuracy"],
                "train_time":   trial["train_time"],
                "config":       json.dumps(trial["config"]),
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   type=str, required=True)
    parser.add_argument("--model",     type=str, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--seed",      type=int, required=True)
    args = parser.parse_args()
    run_single(args.dataset, args.model, args.optimizer, args.seed)
