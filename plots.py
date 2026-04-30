"""Auswertung der HPO-Ergebnisse: Plots."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(_PROJECT_DIR, "results")
PLOTS_DIR    = os.path.join(RESULTS_DIR,  "plots")
LATEX_DIR    = os.path.join(RESULTS_DIR,  "latex")
GFX_DIR      = os.path.join(_PROJECT_DIR, "gfx")

RUNS_FILE   = os.path.join(RESULTS_DIR, "runs.csv")
TRIALS_FILE = os.path.join(RESULTS_DIR, "trials.csv")

PALETTE = {
    "bo":                 "#4C72B0",
    "acrs":               "#DD8452",
    "acrs_normal":        "#55A868",
    "acrs_shrink":        "#C44E52",
    "acrs_normal_shrink": "#8172B2",
    "rs":                 "#937860",
    "cmaes":              "#DA8BC3",
}
LABELS = {
    "bo":                 "BO (TPE)",
    "acrs":               "ACRS",
    "acrs_normal":        "ACRS (Normal)",
    "acrs_shrink":        "ACRS (Shrink)",
    "acrs_normal_shrink": "ACRS (Normal+Shrink)",
    "rs":                 "Random Search",
    "cmaes":              "CMA-ES",
}

ACRS_VARIANTS = ["acrs", "acrs_normal", "acrs_shrink", "acrs_normal_shrink"]
ACRS_BEST     = "acrs_shrink"
_ACRS_OTHERS  = [v for v in ACRS_VARIANTS if v != ACRS_BEST]


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt runs.csv und trials.csv."""
    runs   = pd.read_csv(RUNS_FILE)
    trials = pd.read_csv(TRIALS_FILE)
    return runs, trials


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[report] Saved: {path}")


def plot_convergence(trials: pd.DataFrame):
    """Best-so-far Val-AUROC vs. Trial-Nummer, gemittelt über Seeds (±1 Std).

    2 Zeilen (MLP, RF) × N Spalten (Datasets), eine Figure gesamt.
    """
    trials   = trials[~trials["optimizer"].isin(_ACRS_OTHERS)]
    models   = sorted(trials["model"].unique())
    datasets = sorted(trials["dataset"].unique())

    fig, axes = plt.subplots(len(models), len(datasets), figsize=(6 * len(datasets), 5 * len(models)))

    for row, model in enumerate(models):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            df = trials[(trials["model"] == model) & (trials["dataset"] == dataset)]

            for optimizer in sorted(df["optimizer"].unique()):
                df_opt = df[df["optimizer"] == optimizer]

                curves = []
                for seed in df_opt["seed"].unique():
                    df_seed = df_opt[df_opt["seed"] == seed].sort_values("trial_nr")
                    curves.append(df_seed["val_auroc"].cummax().values)

                min_len = min(len(c) for c in curves)
                mat  = np.array([c[:min_len] for c in curves])
                mean = mat.mean(axis=0)
                std  = mat.std(axis=0)
                x    = np.arange(1, min_len + 1)

                ax.plot(x, mean, label=LABELS[optimizer], color=PALETTE[optimizer])
                ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=PALETTE[optimizer])

            ax.set_title(f"{model.upper()} — {dataset}")
            ax.set_xlabel("Trial")
            ax.set_ylabel("Best Val-AUROC")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence Curves", fontsize=14)
    fig.tight_layout()
    _save(fig, os.path.join(PLOTS_DIR, "convergence.png"))


def plot_convergence_time(trials: pd.DataFrame):
    """Best-so-far Val-AUROC vs. kumulierte Trainingszeit, gemittelt über Seeds (±1 Std).

    2 Zeilen (MLP, RF) × N Spalten (Datasets), eine Figure gesamt.
    Seed-Kurven werden auf ein gemeinsames Zeitgitter interpoliert, bevor gemittelt wird.
    """
    trials   = trials[~trials["optimizer"].isin(_ACRS_OTHERS)]
    models   = sorted(trials["model"].unique())
    datasets = sorted(trials["dataset"].unique())

    fig, axes = plt.subplots(len(models), len(datasets), figsize=(6 * len(datasets), 5 * len(models)))

    for row, model in enumerate(models):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            df = trials[(trials["model"] == model) & (trials["dataset"] == dataset)]

            for optimizer in sorted(df["optimizer"].unique()):
                df_opt = df[df["optimizer"] == optimizer]

                raw_curves = []
                for seed in df_opt["seed"].unique():
                    df_seed = df_opt[df_opt["seed"] == seed].sort_values("trial_nr")
                    cum_time    = df_seed["train_time"].cumsum().values
                    best_so_far = df_seed["val_auroc"].cummax().values
                    raw_curves.append((cum_time, best_so_far))

                # gemeinsames Zeitgitter bis zur kürzesten Seed-Gesamtzeit (keine Extrapolation)
                t_max  = min(c[0][-1] for c in raw_curves)
                t_grid = np.linspace(0, t_max, 200)

                interp = np.array([np.interp(t_grid, t, v) for t, v in raw_curves])
                mean   = interp.mean(axis=0)
                std    = interp.std(axis=0)

                ax.plot(t_grid, mean, label=LABELS[optimizer], color=PALETTE[optimizer])
                ax.fill_between(t_grid, mean - std, mean + std, alpha=0.2, color=PALETTE[optimizer])

            ax.set_title(f"{model.upper()} — {dataset}")
            ax.set_xlabel("Cumulative Trial Training Time (s)")
            ax.set_ylabel("Best Val-AUROC")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence over Time", fontsize=14)
    fig.tight_layout()
    _save(fig, os.path.join(PLOTS_DIR, "convergence_time.png"))


def plot_final_comparison(runs: pd.DataFrame, trials: pd.DataFrame):
    """Boxplots: bestes Val-AUROC und Test-AUROC pro Verfahren, gepooled über Datensätze.

    1 Zeile × 2 Spalten (MLP, RF); jeder Subplot zeigt alle Verfahren nebeneinander.
    Jede Box fasst Seeds × Datensätze zusammen (50 Punkte bei 10 Seeds × 5 Datasets).
    """
    trials     = trials[~trials["optimizer"].isin(_ACRS_OTHERS)]
    runs       = runs[~runs["method"].isin(_ACRS_OTHERS)]
    models     = sorted(trials["model"].unique())
    optimizers = sorted(trials["optimizer"].unique())

    for metric, title, source in [
        ("val_auroc",  "Val-AUROC",  "trials"),
        ("test_auroc", "Test-AUROC", "runs"),
    ]:
        fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5))

        for col, model in enumerate(models):
            ax = axes[col]

            if source == "trials":
                df   = trials[trials["model"] == model]
                best = df.groupby(["optimizer", "seed", "dataset"])["val_auroc"].max().reset_index()
                data = [best[best["optimizer"] == o]["val_auroc"].values for o in optimizers]
            else:
                df   = runs[runs["model_type"] == model]
                data = [df[df["method"] == o]["test_auroc"].values for o in optimizers]

            labels = [LABELS[o] for o in optimizers]
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for patch, o in zip(bp["boxes"], optimizers):
                patch.set_facecolor(PALETTE[o])
                patch.set_alpha(0.7)

            ax.set_title(model.upper())
            ax.set_ylabel(title)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Final Performance — {title}", fontsize=14)
        fig.tight_layout()
        _save(fig, os.path.join(PLOTS_DIR, f"comparison_{metric}.png"))


def plot_acrs_ablation(runs: pd.DataFrame, trials: pd.DataFrame):
    """Vergleich der vier ACRS-Varianten: Boxplots gepooled über Seeds × Datensätze.

    1 Zeile × 2 Spalten (MLP, RF), je eine Figure für Val und Test.
    Jede Box fasst 10 Seeds × 5 Datasets = 50 Punkte zusammen.
    """
    variants = [v for v in ACRS_VARIANTS if v in trials["optimizer"].unique()]
    models   = sorted(trials["model"].unique())

    for metric, title, source in [
        ("val_auroc",  "Val-AUROC",  "trials"),
        ("test_auroc", "Test-AUROC", "runs"),
    ]:
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))

        for col, model in enumerate(models):
            ax = axes[col]

            if source == "trials":
                df   = trials[(trials["model"] == model) & (trials["optimizer"].isin(variants))]
                best = df.groupby(["optimizer", "seed", "dataset"])["val_auroc"].max().reset_index()
                data = [best[best["optimizer"] == v]["val_auroc"].values for v in variants]
            else:
                df   = runs[(runs["model_type"] == model) & (runs["method"].isin(variants))]
                data = [df[df["method"] == v]["test_auroc"].values for v in variants]

            labels = [LABELS[v] for v in variants]
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for patch, v in zip(bp["boxes"], variants):
                patch.set_facecolor(PALETTE[v])
                patch.set_alpha(0.7)

            ax.set_title(model.upper())
            ax.set_ylabel(title)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"ACRS Ablation — {title}", fontsize=14)
        fig.tight_layout()
        _save(fig, os.path.join(PLOTS_DIR, f"acrs_ablation_{metric}.png"))


def plot_training_time(runs: pd.DataFrame):
    """Laufzeit pro Methode und Modell, gemittelt über Seeds (±1 Std).

    1 Zeile × N Spalten (Datasets), eine Figure gesamt.
    """
    runs = runs[~runs["method"].isin(_ACRS_OTHERS)]
    agg  = runs.groupby(["dataset", "model_type", "method"])["duration_s"].agg(["mean", "std"]).reset_index()

    datasets   = sorted(agg["dataset"].unique())
    optimizers = sorted(agg["method"].unique())
    models     = sorted(agg["model_type"].unique())

    x     = np.arange(len(models))
    width = 0.8 / len(optimizers)

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5))

    for col, dataset in enumerate(datasets):
        ax    = axes[col]
        df_ds = agg[agg["dataset"] == dataset]

        for i, opt in enumerate(optimizers):
            df_opt = df_ds[df_ds["method"] == opt]
            means  = [df_opt[df_opt["model_type"] == m]["mean"].values[0] for m in models]
            stds   = [df_opt[df_opt["model_type"] == m]["std"].values[0]  for m in models]
            ax.bar(x + i * width, means, width, yerr=stds,
                   label=LABELS[opt], color=PALETTE[opt], alpha=0.85, capsize=4)

        ax.set_xticks(x + width * (len(optimizers) - 1) / 2)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_title(dataset)
        ax.set_ylabel("Total Duration (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Total Run Duration", fontsize=14)
    fig.tight_layout()
    _save(fig, os.path.join(PLOTS_DIR, "training_time.png"))


def plot_energy(runs: pd.DataFrame):
    """Energieverbrauch pro Methode und Modell, gemittelt über Seeds (±1 Std).

    1 Zeile × N Spalten (Datasets), eine Figure gesamt.
    """
    runs = runs[~runs["method"].isin(_ACRS_OTHERS)]
    agg  = runs.groupby(["dataset", "model_type", "method"])["energy_kwh"].agg(["mean", "std"]).reset_index()

    datasets   = sorted(agg["dataset"].unique())
    optimizers = sorted(agg["method"].unique())
    models     = sorted(agg["model_type"].unique())

    x     = np.arange(len(models))
    width = 0.8 / len(optimizers)

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5))

    for col, dataset in enumerate(datasets):
        ax    = axes[col]
        df_ds = agg[agg["dataset"] == dataset]

        for i, opt in enumerate(optimizers):
            df_opt = df_ds[df_ds["method"] == opt]
            means  = [df_opt[df_opt["model_type"] == m]["mean"].values[0] for m in models]
            stds   = [df_opt[df_opt["model_type"] == m]["std"].values[0]  for m in models]
            ax.bar(x + i * width, means, width, yerr=stds,
                   label=LABELS[opt], color=PALETTE[opt], alpha=0.85, capsize=4)

        ax.set_xticks(x + width * (len(optimizers) - 1) / 2)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_title(dataset)
        ax.set_ylabel("Energy Consumption (kWh)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Energy Consumption", fontsize=14)
    fig.tight_layout()
    _save(fig, os.path.join(PLOTS_DIR, "energy.png"))


def generate_latex_tables(runs: pd.DataFrame):
    """Erzeugt drei LaTeX-Tabellen in LATEX_DIR.

    - table_main_results.tex  : Test-AUROC (Mittelwert ± Std) pro Verfahren × Modell × Dataset
    - table_runtime.tex       : Laufzeit in Sekunden (Mittelwert ± Std)
    - table_significance.tex  : Wilcoxon p-Werte paarweise zwischen Verfahren
    """
    from scipy.stats import wilcoxon

    os.makedirs(LATEX_DIR, exist_ok=True)

    runs = runs[~runs["method"].isin(_ACRS_OTHERS)]
    optimizers = sorted(runs["method"].unique())
    models     = sorted(runs["model_type"].unique())
    datasets   = sorted(runs["dataset"].unique())

    def fmt(mean, std):
        return f"{mean:.3f} $\\pm$ {std:.3f}"

    def tex(s):
        return s.replace("_", "\\_")

    # Tabelle 1: Test-AUROC (Bestwert pro Zeile fettgedruckt)
    with open(os.path.join(LATEX_DIR, "table_main_results.tex"), "w") as f:
        col_spec = "ll" + "r" * len(optimizers)
        headers  = " & ".join(["Dataset", "Modell"] + [LABELS[o] for o in optimizers])
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n{headers} \\\\\n\\midrule\n")
        for dataset in datasets:
            for model in models:
                means = {opt: runs[(runs["method"] == opt) & (runs["model_type"] == model) & (runs["dataset"] == dataset)]["test_auroc"].mean()
                         for opt in optimizers}
                stds  = {opt: runs[(runs["method"] == opt) & (runs["model_type"] == model) & (runs["dataset"] == dataset)]["test_auroc"].std()
                         for opt in optimizers}
                best  = max(means, key=means.get)
                row   = [tex(dataset), model.upper()]
                for opt in optimizers:
                    cell = fmt(means[opt], stds[opt])
                    row.append(f"\\textbf{{{cell}}}" if opt == best else cell)
                f.write(" & ".join(row) + " \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[report] Saved: {os.path.join(LATEX_DIR, 'table_main_results.tex')}")

    # Tabelle 2: Laufzeit
    with open(os.path.join(LATEX_DIR, "table_runtime.tex"), "w") as f:
        col_spec = "ll" + "r" * len(optimizers)
        headers  = " & ".join(["Dataset", "Modell"] + [LABELS[o] for o in optimizers])
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n{headers} \\\\\n\\midrule\n")
        for dataset in datasets:
            for model in models:
                means = {opt: runs[(runs["method"] == opt) & (runs["model_type"] == model) & (runs["dataset"] == dataset)]["duration_s"].mean()
                         for opt in optimizers}
                stds  = {opt: runs[(runs["method"] == opt) & (runs["model_type"] == model) & (runs["dataset"] == dataset)]["duration_s"].std()
                         for opt in optimizers}
                best  = min(means, key=means.get)
                row   = [tex(dataset), model.upper()]
                for opt in optimizers:
                    cell = fmt(means[opt], stds[opt])
                    row.append(f"\\textbf{{{cell}}}" if opt == best else cell)
                f.write(" & ".join(row) + " \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[report] Saved: {os.path.join(LATEX_DIR, 'table_runtime.tex')}")

    # Tabelle 3: Wilcoxon-Signifikanz (paarweise, getrennt pro Modell)
    with open(os.path.join(LATEX_DIR, "table_significance.tex"), "w") as f:
        for model in models:
            f.write(f"\\textbf{{{model.upper()}}}\\\\\n")
            col_spec = "l" + "r" * len(optimizers)
            headers  = " & ".join([""] + [LABELS[o] for o in optimizers])
            f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n{headers} \\\\\n\\midrule\n")
            df_m = runs[runs["model_type"] == model]
            for i, o1 in enumerate(optimizers):
                row = [LABELS[o1]]
                for j, o2 in enumerate(optimizers):
                    if j == i:
                        row.append("—")
                    elif j < i:
                        # unteres Dreieck leer lassen
                        row.append("")
                    else:
                        # gepoolte Paare: gleicher seed × gleicher dataset
                        merged = df_m[df_m["method"] == o1][["seed", "dataset", "test_auroc"]].merge(
                            df_m[df_m["method"] == o2][["seed", "dataset", "test_auroc"]],
                            on=["seed", "dataset"], suffixes=("_a", "_b")
                        )
                        _, p = wilcoxon(merged["test_auroc_a"], merged["test_auroc_b"])
                        mark = "*" if p < 0.05 else ""
                        row.append(f"{p:.3f}{mark}")
                f.write(" & ".join(row) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\bigskip\n\n")
    print(f"[report] Saved: {os.path.join(LATEX_DIR, 'table_significance.tex')}")


def main():
    runs, trials = _load_data()
    plot_convergence(trials)
    plot_convergence_time(trials)
    plot_final_comparison(runs, trials)
    plot_acrs_ablation(runs, trials)
    plot_training_time(runs)
    plot_energy(runs)
    generate_latex_tables(runs)


if __name__ == "__main__":
    main()
