"""Outer-Loop: alle Experimente ausführen und Ergebnisse in CSV speichern."""

import csv
import json
import os
import time

from codecarbon import EmissionsTracker

from config import SEEDS, DATASETS
from data import load_data
from optimize import bayesian_search, acrs_search
from train import eval_test_mlp, eval_test_rf

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUNS_FILE   = os.path.join(RESULTS_DIR, "runs.csv")
TRIALS_FILE = os.path.join(RESULTS_DIR, "trials.csv")

RUNS_FIELDNAMES = ["dataset", "seed", "method", "model_type",
                   "val_auroc", "val_accuracy", "test_auroc", "test_accuracy",
                   "duration_s", "energy_kwh", "best_config"]

TRIALS_FIELDNAMES = ["trial_nr", "dataset", "model", "optimizer", "seed",
                     "val_auroc", "val_accuracy", "train_time", "config"]

# 4 Experimente pro Dataset × Seed
EXPERIMENTS = [
    ("bo",   "mlp"),
    ("bo",   "rf"),
    ("acrs", "mlp"),
    ("acrs", "rf"),
]


def _run(method: str, model_type: str, data: dict, seed: int) -> dict:
    """Führt einen einzelnen Run durch und gibt die Ergebnisse als Dict zurück."""
    tracker = EmissionsTracker(save_to_file=False, logging_logger=None)
    tracker.start()
    t0 = time.time()

    if method == "bo":
        best_config, val_auroc, trial_history = bayesian_search(data, model_type=model_type, seed=seed)
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
        "val_auroc":     round(float(val_auroc),    4),
        "val_accuracy":  round(float(val_accuracy), 4),
        "test_auroc":    round(float(test_auroc),   4),
        "test_accuracy": round(float(test_accuracy), 4),
        "duration_s":    round(duration,     1),
        "energy_kwh":    round(energy_kwh,   6),
        "best_config":   json.dumps(best_config),
        "trial_history": trial_history,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(RUNS_FILE, "w", newline="") as runs_f, \
         open(TRIALS_FILE, "w", newline="") as trials_f:

        runs_writer   = csv.DictWriter(runs_f,   fieldnames=RUNS_FIELDNAMES)
        trials_writer = csv.DictWriter(trials_f, fieldnames=TRIALS_FIELDNAMES)
        runs_writer.writeheader()
        trials_writer.writeheader()

        for dataset in DATASETS:
            for seed in SEEDS:
                data = load_data(seed=seed, dataset=dataset)
                print(f"\n=== {dataset} | seed={seed} ===")

                for method, model_type in EXPERIMENTS:
                    label = f"{method.upper()} {model_type.upper()}"
                    print(f"  {label} ...", end=" ", flush=True)

                    result = _run(method, model_type, data, seed)
                    trial_history = result.pop("trial_history")

                    print(f"val={result['val_auroc']:.4f}  "
                          f"test={result['test_auroc']:.4f}  "
                          f"{result['duration_s']:.0f}s")

                    # Zeile in runs.csv schreiben
                    runs_writer.writerow({
                        "dataset":    dataset,
                        "seed":       seed,
                        "method":     method,
                        "model_type": model_type,
                        **result,
                    })
                    runs_f.flush()

                    # Zeilen in trials.csv schreiben
                    for trial in trial_history:
                        trials_writer.writerow({
                            "trial_nr":    trial["trial_nr"],
                            "dataset":     dataset,
                            "model":       model_type,
                            "optimizer":   method,
                            "seed":        seed,
                            "val_auroc":   trial["val_auroc"],
                            "val_accuracy": trial["val_accuracy"],
                            "train_time":  trial["train_time"],
                            "config":      json.dumps(trial["config"]),
                        })
                    trials_f.flush()


if __name__ == "__main__":
    main()
