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

# konsistente Farben für BO und ACRS
PALETTE = {"bo": "#4C72B0", "acrs": "#DD8452"}
LABELS  = {"bo": "BO (TPE)", "acrs": "ACRS"}


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

    2 Zeilen (MLP, RF) × 3 Spalten (Datasets), eine Figure gesamt.
    """
    models   = sorted(trials["model"].unique())
    datasets = sorted(trials["dataset"].unique())

    # 2×3 Grid: Zeilen = Modelle, Spalten = Datasets
    fig, axes = plt.subplots(len(models), len(datasets), figsize=(6 * len(datasets), 5 * len(models)))

    for row, model in enumerate(models):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            # nur Zeilen für dieses Dataset + Modell
            df = trials[(trials["model"] == model) & (trials["dataset"] == dataset)]

            for optimizer in sorted(df["optimizer"].unique()):
                df_opt = df[df["optimizer"] == optimizer]

                # best-so-far Kurve pro Seed berechnen (cummax = laufendes Maximum)
                curves = []
                for seed in df_opt["seed"].unique():
                    df_seed = df_opt[df_opt["seed"] == seed].sort_values("trial_nr")
                    curves.append(df_seed["val_auroc"].cummax().values)

                # auf kürzeste Kurve kürzen damit BO und ACRS dieselbe X-Achse haben
                min_len = min(len(c) for c in curves)
                # Matrix (3 Seeds × min_len Trials) → Mittelwert und Std pro Trial
                mat  = np.array([c[:min_len] for c in curves])
                mean = mat.mean(axis=0)
                std  = mat.std(axis=0)
                x    = np.arange(1, min_len + 1)

                # Mittellinie + schattierte ±1 Std Fläche
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


def plot_final_comparison(runs: pd.DataFrame, trials: pd.DataFrame):
    """Boxplots: bestes Val-AUROC und Test-AUROC pro Seed, BO vs. ACRS.

    2 Zeilen (MLP, RF) × 3 Spalten (Datasets), je eine Figure für Val und Test.
    """
    models   = sorted(trials["model"].unique())
    datasets = sorted(trials["dataset"].unique())

    # Schleife läuft zweimal: einmal für Val-AUROC (aus trials), einmal für Test-AUROC (aus runs)
    for metric, title, source in [
        ("val_auroc",  "Val-AUROC",  "trials"),
        ("test_auroc", "Test-AUROC", "runs"),
    ]:
        # 2×3 Grid: Zeilen = Modelle, Spalten = Datasets
        fig, axes = plt.subplots(len(models), len(datasets), figsize=(6 * len(datasets), 5 * len(models)))

        for row, model in enumerate(models):
            for col, dataset in enumerate(datasets):
                ax = axes[row, col]

                if source == "trials":
                    # bestes Val-AUROC pro Optimizer + Seed (Maximum über alle Trials)
                    df = trials[(trials["model"] == model) & (trials["dataset"] == dataset)]
                    best = df.groupby(["optimizer", "seed"])["val_auroc"].max().reset_index()
                else:
                    # Test-AUROC aus runs; Spalten umbenennen für einheitliche Logik danach
                    df = runs[(runs["model_type"] == model) & (runs["dataset"] == dataset)]
                    best = df[["method", "seed", "test_auroc"]].rename(columns={
                        "method":     "optimizer",
                        "test_auroc": "val_auroc",
                    })

                ## Boxsplots zeichnen
                # pro Optimizer die 3 Seed-Werte als Liste sammeln
                optimizers = sorted(best["optimizer"].unique())
                data   = [best[best["optimizer"] == o]["val_auroc"].values for o in optimizers]
                labels = [LABELS[o] for o in optimizers]

                # patch_artist=True erlaubt das Einfärben der Boxen
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                for patch, o in zip(bp["boxes"], optimizers):
                    patch.set_facecolor(PALETTE[o])
                    patch.set_alpha(0.7)

                ax.set_title(f"{model.upper()} — {dataset}")
                ax.set_ylabel(title)
                ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Final Performance — {title}", fontsize=14)
        fig.tight_layout()
        _save(fig, os.path.join(PLOTS_DIR, f"comparison_{metric}.png"))


def plot_training_time(runs: pd.DataFrame):
    """Kumulierte Laufzeit pro Methode und Modell, gemittelt über Seeds (±1 Std).

    1 Zeile × 3 Spalten (Datasets), eine Figure gesamt.
    """
    # Mittelwert und Std der Laufzeit über die 3 Seeds berechnen
    agg = runs.groupby(["dataset", "model_type", "method"])["duration_s"].agg(["mean", "std"]).reset_index()

    datasets   = sorted(agg["dataset"].unique())
    optimizers = sorted(agg["method"].unique())
    models     = sorted(agg["model_type"].unique())

    # Positionen der Balkengruppen auf der X-Achse (eine pro Modell)
    x     = np.arange(len(models))
    # Breite pro Balken: 0.8 aufgeteilt auf alle Optimizer
    width = 0.8 / len(optimizers)

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5))

    for col, dataset in enumerate(datasets):
        ax    = axes[col]
        df_ds = agg[agg["dataset"] == dataset]

        # pro Optimizer einen verschobenen Balken pro Modell zeichnen
        for i, opt in enumerate(optimizers):
            df_opt = df_ds[df_ds["method"] == opt]
            means  = [df_opt[df_opt["model_type"] == m]["mean"].values[0] for m in models]
            stds   = [df_opt[df_opt["model_type"] == m]["std"].values[0]  for m in models]
            # i * width verschiebt den Balken nach rechts neben den vorherigen
            ax.bar(x + i * width, means, width, yerr=stds,
                   label=LABELS[opt], color=PALETTE[opt], alpha=0.85, capsize=4)

        # X-Achsen-Labels mittig zwischen den zwei Balken einer Gruppe
        ax.set_xticks(x + width * (len(optimizers) - 1) / 2)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_title(dataset)
        ax.set_ylabel("Total Duration (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Total Run Duration — BO vs. ACRS", fontsize=14)
    fig.tight_layout()
    _save(fig, os.path.join(PLOTS_DIR, "training_time.png"))


def plot_energy(runs: pd.DataFrame):
    """Energieverbrauch pro Methode und Modell, gemittelt über Seeds (±1 Std).

    1 Zeile × 3 Spalten (Datasets), eine Figure gesamt.
    """
    # Mittelwert und Std des Energieverbrauchs über die 3 Seeds berechnen
    agg = runs.groupby(["dataset", "model_type", "method"])["energy_kwh"].agg(["mean", "std"]).reset_index()

    datasets   = sorted(agg["dataset"].unique())
    optimizers = sorted(agg["method"].unique())
    models     = sorted(agg["model_type"].unique())

    # Positionen der Balkengruppen auf der X-Achse (eine pro Modell)
    x     = np.arange(len(models))
    # Breite pro Balken: 0.8 aufgeteilt auf alle Optimizer
    width = 0.8 / len(optimizers)

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5))

    for col, dataset in enumerate(datasets):
        ax    = axes[col]
        df_ds = agg[agg["dataset"] == dataset]

        # pro Optimizer einen verschobenen Balken pro Modell zeichnen
        for i, opt in enumerate(optimizers):
            df_opt = df_ds[df_ds["method"] == opt]
            means  = [df_opt[df_opt["model_type"] == m]["mean"].values[0] for m in models]
            stds   = [df_opt[df_opt["model_type"] == m]["std"].values[0]  for m in models]
            # i * width verschiebt den Balken nach rechts neben den vorherigen
            ax.bar(x + i * width, means, width, yerr=stds,
                   label=LABELS[opt], color=PALETTE[opt], alpha=0.85, capsize=4)

        # X-Achsen-Labels mittig zwischen den zwei Balken einer Gruppe
        ax.set_xticks(x + width * (len(optimizers) - 1) / 2)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_title(dataset)
        ax.set_ylabel("Energy Consumption (kWh)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Energy Consumption — BO vs. ACRS", fontsize=14)
    fig.tight_layout()
    _save(fig, os.path.join(PLOTS_DIR, "energy.png"))


def plot_efficiency(trials: pd.DataFrame):
    """Kumulierte Trainingszeit vs. best-so-far Val-AUROC, pro Seed eine Linie.

    2 Zeilen (MLP, RF) × 3 Spalten (Datasets), eine Figure gesamt.
    """
    models   = sorted(trials["model"].unique())
    datasets = sorted(trials["dataset"].unique())

    # 2x3 Grid: Zeilen = Modelle, Spalten = Datasets
    fig, axes = plt.subplots(len(models), len(datasets),
                             figsize=(6 * len(datasets), 5 * len(models)))

    for row, model in enumerate(models):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            # nur Zeilen für dieses Dataset + Modell
            df = trials[(trials["model"] == model) & (trials["dataset"] == dataset)]

            for optimizer in sorted(df["optimizer"].unique()):
                df_opt = df[df["optimizer"] == optimizer]
                # Label nur beim ersten Seed setzen, sonst erscheint jede Linie in der Legende
                first  = True

                for seed in df_opt["seed"].unique():
                    df_seed = df_opt[df_opt["seed"] == seed].sort_values("trial_nr")

                    # kumulierte Zeit und laufendes Maximum berechnen
                    cum_time    = df_seed["train_time"].cumsum().values
                    best_so_far = df_seed["val_auroc"].cummax().values

                    # Label nur beim ersten Seed setzen
                    ax.plot(cum_time, best_so_far,
                            color=PALETTE[optimizer], alpha=0.4, linewidth=1.2,
                            label=LABELS[optimizer] if first else None)
                    first = False

            ax.set_title(f"{model.upper()} — {dataset}")
            ax.set_xlabel("Cumulative Training Time (s)")
            ax.set_ylabel("Best Val-AUROC")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Efficiency — BO vs. ACRS", fontsize=14)
    fig.tight_layout()
    _save(fig, os.path.join(PLOTS_DIR, "efficiency.png"))


def main():
    runs, trials = _load_data()
    plot_convergence(trials)
    plot_final_comparison(runs, trials)
    plot_training_time(runs)
    plot_energy(runs)
    plot_efficiency(trials)


if __name__ == "__main__":
    main()

