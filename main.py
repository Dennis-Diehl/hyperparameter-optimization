"""Startet alle Experimente parallel und führt Ergebnisse zusammen."""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import SEEDS, DATASETS

EXPERIMENTS = [
    ("bo",    "mlp"),
    ("bo",    "rf"),
    ("acrs",  "mlp"),
    ("acrs",  "rf"),
    ("rs",    "mlp"),
    ("rs",    "rf"),
    ("cmaes", "mlp"),
    ("cmaes", "rf"),
]
MAX_WORKERS = 6

_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
PARTIAL_DIR = os.path.join(RESULTS_DIR, "partial")
RUNS_FILE   = os.path.join(RESULTS_DIR, "runs.csv")
TRIALS_FILE = os.path.join(RESULTS_DIR, "trials.csv")

RUNS_FIELDNAMES   = ["dataset", "seed", "method", "model_type",
                     "val_auroc", "val_accuracy", "test_auroc", "test_accuracy",
                     "duration_s", "energy_kwh", "best_config"]
TRIALS_FIELDNAMES = ["trial_nr", "dataset", "model", "optimizer", "seed",
                     "val_auroc", "val_accuracy", "train_time", "config"]


def _run_subprocess(args: tuple) -> tuple:
    """Startet run_single.py als Subprozess für eine Kombination."""
    dataset, model_type, method, seed = args
    subprocess.run(
        [sys.executable, os.path.join(_HERE, "run_single.py"),
         "--dataset",   dataset,
         "--model",     model_type,
         "--optimizer", method,
         "--seed",      str(seed)],
        check=True,
    )
    return (dataset, model_type, method, seed)


def merge_results():
    """Führt alle Partial-CSVs zu runs.csv und trials.csv zusammen."""
    runs_files   = sorted(glob.glob(os.path.join(PARTIAL_DIR, "runs_*.csv")))
    trials_files = sorted(glob.glob(os.path.join(PARTIAL_DIR, "trials_*.csv")))

    with open(RUNS_FILE, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=RUNS_FIELDNAMES)
        writer.writeheader()
        for f in runs_files:
            with open(f) as inp:
                for row in csv.DictReader(inp):
                    writer.writerow(row)

    with open(TRIALS_FILE, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=TRIALS_FIELDNAMES)
        writer.writeheader()
        for f in trials_files:
            with open(f) as inp:
                for row in csv.DictReader(inp):
                    writer.writerow(row)

    print(f"\n{len(runs_files)} Runs zusammengeführt → runs.csv + trials.csv")
    shutil.rmtree(PARTIAL_DIR)
    print("partial/ gelöscht.")


def run_all():
    """Startet alle Kombinationen parallel."""
    combinations = [
        (dataset, model_type, method, seed)
        for dataset        in DATASETS
        for seed           in SEEDS
        for method, model_type in EXPERIMENTS
    ]
    total = len(combinations)
    print(f"Starte {total} Runs mit {MAX_WORKERS} parallelen Prozessen...\n")
    os.makedirs(PARTIAL_DIR, exist_ok=True)

    completed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_subprocess, c): c for c in combinations}
        for future in as_completed(futures):
            dataset, model_type, method, seed = future.result()
            completed += 1
            print(f"[{completed}/{total}] fertig | {dataset} {model_type} {method} seed={seed}")

    merge_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   type=str)
    parser.add_argument("--model",     type=str)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--seed",      type=int)
    args = parser.parse_args()

    if args.dataset:
        from run_single import run_single
        run_single(args.dataset, args.model, args.optimizer, args.seed)
        merge_results()
    else:
        run_all()
