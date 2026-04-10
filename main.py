"""Outer-Loop: alle Experimente ausführen und Ergebnisse in CSV speichern."""

import csv
import json
import os
import time

from config import SEEDS, DATASETS
from data import load_data
from optimize import bayesian_search, acrs_search
from train import eval_test_mlp, eval_test_rf

RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.csv")

FIELDNAMES = ["dataset", "seed", "method", "model_type",
              "val_auroc", "test_auroc", "duration_s", "best_config"]

# 4 Experimente pro Dataset × Seed
EXPERIMENTS = [
    ("bo",   "mlp"),
    ("bo",   "rf"),
    ("acrs", "mlp"),
    ("acrs", "rf"),
]


def _run(method: str, model_type: str, data: dict, seed: int) -> dict:
    """Führt einen einzelnen Run durch und gibt die Ergebnisse als Dict zurück."""
    t0 = time.time()
    if method == "bo":
        best_config, val_auroc = bayesian_search(data, model_type=model_type, seed=seed)
    else:
        best_config, val_auroc = acrs_search(data, model_type=model_type, seed=seed)
    duration = time.time() - t0

    if model_type == "mlp":
        test_auroc = eval_test_mlp(best_config, data)
    else:
        test_auroc = eval_test_rf(best_config, data)

    return {
        "val_auroc":   round(val_auroc,  4),
        "test_auroc":  round(test_auroc, 4),
        "duration_s":  round(duration,   1),
        "best_config": json.dumps(best_config),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for dataset in DATASETS:
            for seed in SEEDS:
                data = load_data(seed=seed, dataset=dataset)
                print(f"\n=== {dataset} | seed={seed} ===")

                for method, model_type in EXPERIMENTS:
                    label = f"{method.upper()} {model_type.upper()}"
                    print(f"  {label} ...", end=" ", flush=True)

                    result = _run(method, model_type, data, seed)
                    print(f"val={result['val_auroc']:.4f}  "
                          f"test={result['test_auroc']:.4f}  "
                          f"{result['duration_s']:.0f}s")

                    writer.writerow({
                        "dataset":    dataset,
                        "seed":       seed,
                        "method":     method,
                        "model_type": model_type,
                        **result,
                    })
                    f.flush()  # Zwischenergebnisse sofort schreiben


if __name__ == "__main__":
    main()
