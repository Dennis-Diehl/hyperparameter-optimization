"""Auswertung der HPO-Ergebnisse: Plots."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(_PROJECT_DIR, "results")
PLOTS_DIR    = os.path.join(RESULTS_DIR,  "plots")
LATEX_DIR    = os.path.join(RESULTS_DIR,  "latex")
GFX_DIR      = os.path.join(_PROJECT_DIR, "Thesis-Ausarbeitung", "gfx")

RUNS_FILE   = os.path.join(RESULTS_DIR, "runs.csv")
TRIALS_FILE = os.path.join(RESULTS_DIR, "trials.csv")

PALETTE = {
    "bo":                 "#0072B2",  # Okabe-Ito Blue
    "acrs_shrink":        "#D55E00",  # Okabe-Ito Vermilion
    "rs":                 "#009E73",  # Okabe-Ito Bluish Green
    "cmaes":              "#CC79A7",  # Okabe-Ito Reddish Purple
    "acrs":               "#E69F00",  # Okabe-Ito Orange
    "acrs_normal":        "#56B4E9",  # Okabe-Ito Sky Blue
    "acrs_normal_shrink": "#F0E442",  # Okabe-Ito Yellow
}
LABELS = {
    "bo":                 "BO (TPE)",
    "acrs":               "ACRS",
    "acrs_normal":        "ACRS (Normal)",
    "acrs_shrink":        "ACRS (Shrink)",
    "acrs_normal_shrink": "ACRS (Normal+Shrink)",
    "rs":                 "RS",
    "cmaes":              "CMA-ES",
}

ACRS_VARIANTS = ["acrs", "acrs_normal", "acrs_shrink", "acrs_normal_shrink"]
ACRS_BEST     = "acrs_shrink"
_ACRS_OTHERS  = [v for v in ACRS_VARIANTS if v != ACRS_BEST]

# Ab der Ablation repräsentiert ACRS (Shrink) die Methode; im Haupttext heißt sie nur "ACRS"
LABELS_MAIN = {**LABELS, ACRS_BEST: "ACRS"}

# Repräsentative Datasets für den gekürzten Haupttext-Ausschnitt der Convergence-Plots;
# das volle Raster über alle Datasets landet im Appendix.
MAIN_DATASETS = ["adult_income", "german_credit"]


def _select_bold(means: dict, stds: dict, higher_is_better: bool, decimals: int) -> set:
    """Bestimmt welche Einträge einer Tabellenzeile fettgedruckt werden.

    Primärkriterium: bester gerundeter Mean.
    Sekundärkriterium bei Gleichstand: kleinste gerundete Std.
    Bei vollständigem Gleichstand: alle gleichwertigen Einträge.
    """
    r_means = {k: round(v, decimals) for k, v in means.items()}
    r_stds  = {k: round(v, decimals) for k, v in stds.items()}
    best_mean = max(r_means.values()) if higher_is_better else min(r_means.values())
    tied      = [k for k, v in r_means.items() if v == best_mean]
    if len(tied) == 1:
        return set(tied)
    tied_stds = {k: r_stds[k] for k in tied}
    best_std  = min(tied_stds.values())
    return {k for k, v in tied_stds.items() if v == best_std}


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


def plot_convergence(trials: pd.DataFrame, datasets: list[str] | None = None,
                     filename: str = "convergence.png", show_std: bool = True):
    """Best-so-far Val-AUROC vs. Trial-Nummer, gemittelt über Seeds (±1 Std).

    2 Zeilen (MLP, RF) × N Spalten (Datasets), eine Figure gesamt.
    datasets=None erzeugt das volle Raster (Appendix); eine Teilmenge erzeugt
    den gekürzten Haupttext-Ausschnitt.
    show_std=False blendet das Standardabweichungs-Band aus (übersichtlicher
    Haupttext); die Streuung wird über die Appendix-Figur und die Tabellen abgedeckt.
    """
    trials   = trials[~trials["optimizer"].isin(_ACRS_OTHERS)]
    models   = sorted(trials["model"].unique())
    datasets = datasets or sorted(trials["dataset"].unique())

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

                ax.plot(x, mean, label=LABELS_MAIN[optimizer], color=PALETTE[optimizer])
                if show_std:
                    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=PALETTE[optimizer])

            ax.set_title(f"{model.upper()} — {dataset.replace('_', ' ').title()}")
            ax.set_xlabel("Trial")
            ax.set_ylabel("Best Val-AUROC")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence Curves", fontsize=14)
    fig.tight_layout()
    _save(fig, os.path.join(PLOTS_DIR, filename))


def plot_convergence_time(trials: pd.DataFrame, datasets: list[str] | None = None,
                          filename: str = "convergence_time.png", show_std: bool = True):
    """Best-so-far Val-AUROC vs. kumulierte Trainingszeit, gemittelt über Seeds (±1 Std).

    2 Zeilen (MLP, RF) × N Spalten (Datasets), eine Figure gesamt.
    Seed-Kurven werden auf ein gemeinsames Zeitgitter interpoliert, bevor gemittelt wird.
    datasets=None erzeugt das volle Raster (Appendix); eine Teilmenge erzeugt
    den gekürzten Haupttext-Ausschnitt.
    show_std=False blendet das Standardabweichungs-Band aus (übersichtlicher
    Haupttext); die Streuung wird über die Appendix-Figur und die Tabellen abgedeckt.
    """
    trials   = trials[~trials["optimizer"].isin(_ACRS_OTHERS)]
    models   = sorted(trials["model"].unique())
    datasets = datasets or sorted(trials["dataset"].unique())

    fig, axes = plt.subplots(len(models), len(datasets), figsize=(6 * len(datasets), 5 * len(models)))

    for row, model in enumerate(models):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            df = trials[(trials["model"] == model) & (trials["dataset"] == dataset)]

            for optimizer in sorted(df["optimizer"].unique()):
                df_opt = df[df["optimizer"] == optimizer]

                cum_times   = []
                best_curves = []
                for seed in df_opt["seed"].unique():
                    df_seed = df_opt[df_opt["seed"] == seed].sort_values("trial_nr")
                    cum_times.append(df_seed["train_time"].cumsum().values)
                    best_curves.append(df_seed["val_auroc"].cummax().values)

                # pro Trial über die Seeds mitteln (gleiche Trial-Zahl je Seed): die Kurve
                # endet damit bei der mittleren kumulativen Trainingszeit, ohne Clipping
                L = min(len(c) for c in best_curves)
                x    = np.array([c[:L] for c in cum_times]).mean(axis=0)
                best = np.array([c[:L] for c in best_curves])
                mean = best.mean(axis=0)
                std  = best.std(axis=0)

                ax.plot(x, mean, label=LABELS_MAIN[optimizer], color=PALETTE[optimizer])
                if show_std:
                    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=PALETTE[optimizer])

            ax.set_title(f"{model.upper()} — {dataset.replace('_', ' ').title()}")
            ax.set_xlabel("Cumulative Trial Training Time (s)")
            ax.set_ylabel("Best Val-AUROC")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence over Time", fontsize=14)
    fig.tight_layout()
    _save(fig, os.path.join(PLOTS_DIR, filename))


def generate_final_performance_table(runs: pd.DataFrame, trials: pd.DataFrame):
    """Erzeugt LaTeX-Tabelle: Val-AUROC und Test-AUROC (mean ± std) pro Methode,
    gepoolt über alle Datasets × Seeds, je Zeile ein Modell × Metrik.
    Bester (höchster) Wert pro Zeile fettgedruckt.

    Val-AUROC ist der beste Trial-Wert pro Run (max über Trials), Test-AUROC
    der finale Wert des retrainierten Bestmodells aus runs.csv.
    """
    trials     = trials[~trials["optimizer"].isin(_ACRS_OTHERS)]
    runs       = runs[~runs["method"].isin(_ACRS_OTHERS)]
    optimizers = sorted(runs["method"].unique())
    models     = sorted(runs["model_type"].unique())

    # bestes Val-AUROC pro Run (max über Trials)
    best_val = trials.groupby(["optimizer", "model", "seed", "dataset"])["val_auroc"].max().reset_index()

    os.makedirs(LATEX_DIR, exist_ok=True)

    def _stats(metric: str, model: str):
        if metric == "val_auroc":
            return {o: best_val[(best_val["optimizer"] == o) & (best_val["model"] == model)]["val_auroc"] for o in optimizers}
        return {o: runs[(runs["method"] == o) & (runs["model_type"] == model)]["test_auroc"] for o in optimizers}

    metrics = [("val_auroc", "Val-AUROC"), ("test_auroc", "Test-AUROC")]

    col_spec = "ll" + "c" * len(optimizers)
    headers  = " & ".join(["Modell", "Metric"] + [LABELS_MAIN[o] for o in optimizers])

    with open(os.path.join(LATEX_DIR, "table_final_performance.tex"), "w") as f:
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n{headers} \\\\\n\\midrule\n")
        for m_idx, model in enumerate(models):
            for metric, label in metrics:
                stats = _stats(metric, model)
                means = {o: stats[o].mean() for o in optimizers}
                stds  = {o: stats[o].std()  for o in optimizers}
                bold  = _select_bold(means, stds, higher_is_better=True, decimals=3)
                row   = [model.upper(), label]
                for o in optimizers:
                    cell = f"{means[o]:.3f} $\\pm$ {stds[o]:.3f}"
                    row.append(f"\\textbf{{{cell}}}" if o in bold else cell)
                f.write(" & ".join(row) + " \\\\\n")
            if m_idx < len(models) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"[report] Saved: {os.path.join(LATEX_DIR, 'table_final_performance.tex')}")


def generate_acrs_ablation_table(runs: pd.DataFrame, trials: pd.DataFrame):
    """Erzeugt LaTeX-Tabelle: Test-AUROC (mean ± std) der vier ACRS-Varianten,
    gepooled über alle Datasets × Seeds, getrennt nach Modell (MLP / RF).
    """
    variants = [v for v in ACRS_VARIANTS if v in runs["method"].unique()]
    models   = sorted(runs["model_type"].unique())

    os.makedirs(LATEX_DIR, exist_ok=True)

    col_spec = "l" + "c" * len(variants)
    headers  = " & ".join([""] + [LABELS[v] for v in variants])

    with open(os.path.join(LATEX_DIR, "table_acrs_ablation.tex"), "w") as f:
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n{headers} \\\\\n\\midrule\n")
        for model in models:
            row = [model.upper()]
            stats  = {v: runs[(runs["method"] == v) & (runs["model_type"] == model)]["test_auroc"] for v in variants}
            means  = {v: stats[v].mean() for v in variants}
            stds   = {v: stats[v].std()  for v in variants}
            bold   = _select_bold(means, stds, higher_is_better=True, decimals=3)
            for v in variants:
                cell = f"{means[v]:.3f} $\\pm$ {stds[v]:.3f}"
                row.append(f"\\textbf{{{cell}}}" if v in bold else cell)
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"[report] Saved: {os.path.join(LATEX_DIR, 'table_acrs_ablation.tex')}")


def generate_efficiency_table(runs: pd.DataFrame):
    """Erzeugt LaTeX-Tabelle: Laufzeit (s) und Energieverbrauch (kWh) pro Methode,
    gepoolt über alle Datasets × Seeds, je Zeile ein Modell × Metrik.
    Bester (niedrigster) Wert pro Zeile fettgedruckt.
    """
    runs       = runs[~runs["method"].isin(_ACRS_OTHERS)]
    optimizers = sorted(runs["method"].unique())
    models     = sorted(runs["model_type"].unique())

    os.makedirs(LATEX_DIR, exist_ok=True)

    metrics = [
        ("duration_s", "Duration (s)",  1, "{:.1f}"),
        ("energy_kwh", "Energy (kWh)",  4, "{:.4f}"),
    ]

    col_spec = "ll" + "c" * len(optimizers)
    headers  = " & ".join(["Modell", "Metric"] + [LABELS_MAIN[o] for o in optimizers])

    with open(os.path.join(LATEX_DIR, "table_efficiency.tex"), "w") as f:
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n{headers} \\\\\n\\midrule\n")
        for m_idx, model in enumerate(models):
            for col_name, label, decimals, num_fmt in metrics:
                stats = {o: runs[(runs["method"] == o) & (runs["model_type"] == model)][col_name] for o in optimizers}
                means = {o: stats[o].mean() for o in optimizers}
                stds  = {o: stats[o].std()  for o in optimizers}
                bold  = _select_bold(means, stds, higher_is_better=False, decimals=decimals)
                row   = [model.upper(), label]
                for o in optimizers:
                    cell = f"{num_fmt.format(means[o])} $\\pm$ {num_fmt.format(stds[o])}"
                    row.append(f"\\textbf{{{cell}}}" if o in bold else cell)
                f.write(" & ".join(row) + " \\\\\n")
            if m_idx < len(models) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"[report] Saved: {os.path.join(LATEX_DIR, 'table_efficiency.tex')}")


def generate_latex_tables(runs: pd.DataFrame):
    """Erzeugt zwei LaTeX-Tabellen in LATEX_DIR.

    - table_main_results.tex : Test-AUROC (Mittelwert ± Std) pro Verfahren × Modell × Dataset
    - table_runtime.tex      : Laufzeit in Sekunden (Mittelwert ± Std)
    """
    os.makedirs(LATEX_DIR, exist_ok=True)

    runs = runs[~runs["method"].isin(_ACRS_OTHERS)]
    optimizers = sorted(runs["method"].unique())
    models     = sorted(runs["model_type"].unique())
    datasets   = sorted(runs["dataset"].unique())

    # formatierte Zeile: Mittelwert + Standardabweichung mit drei Nachkommastellen
    def fmt(mean, std):
        return f"{mean:.3f} $\\pm$ {std:.3f}"

    # Formatierte Zeile: Trainingszeit + Standardabweichung mit einer Nachkommastelle
    def fmt1(mean, std):
        return f"{mean:.1f} $\\pm$ {std:.1f}"

    # Datensatznamen lesbar machen: adult_income -> Adult Income
    def pretty(s):
        return s.replace("_", " ").title()

    # Tabelle 1: Test-AUROC (Bestwert pro Zeile fettgedruckt)
    with open(os.path.join(LATEX_DIR, "table_main_results.tex"), "w") as f:
        col_spec = "ll" + "c" * len(optimizers)
        headers  = " & ".join(["Dataset", "Modell"] + [LABELS_MAIN[o] for o in optimizers])
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n{headers} \\\\\n\\midrule\n")
        for i, dataset in enumerate(datasets):
            for model in models:
                means = {opt: runs[(runs["method"] == opt) & (runs["model_type"] == model) & (runs["dataset"] == dataset)]["test_auroc"].mean()
                         for opt in optimizers}
                stds  = {opt: runs[(runs["method"] == opt) & (runs["model_type"] == model) & (runs["dataset"] == dataset)]["test_auroc"].std()
                         for opt in optimizers}
                bold  = _select_bold(means, stds, higher_is_better=True, decimals=3)
                row   = [pretty(dataset), model.upper()]
                for opt in optimizers:
                    cell = fmt(means[opt], stds[opt])
                    row.append(f"\\textbf{{{cell}}}" if opt in bold else cell)
                f.write(" & ".join(row) + " \\\\\n")
            if i < len(datasets) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[report] Saved: {os.path.join(LATEX_DIR, 'table_main_results.tex')}")

    # Tabelle 2: Laufzeit
    with open(os.path.join(LATEX_DIR, "table_runtime.tex"), "w") as f:
        col_spec = "ll" + "c" * len(optimizers)
        headers  = " & ".join(["Dataset", "Modell"] + [LABELS_MAIN[o] for o in optimizers])
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n{headers} \\\\\n\\midrule\n")
        for i, dataset in enumerate(datasets):
            for model in models:
                means = {opt: runs[(runs["method"] == opt) & (runs["model_type"] == model) & (runs["dataset"] == dataset)]["duration_s"].mean()
                         for opt in optimizers}
                stds  = {opt: runs[(runs["method"] == opt) & (runs["model_type"] == model) & (runs["dataset"] == dataset)]["duration_s"].std()
                         for opt in optimizers}
                bold  = _select_bold(means, stds, higher_is_better=False, decimals=1)
                row   = [pretty(dataset), model.upper()]
                for opt in optimizers:
                    cell = fmt1(means[opt], stds[opt])
                    row.append(f"\\textbf{{{cell}}}" if opt in bold else cell)
                f.write(" & ".join(row) + " \\\\\n")
            if i < len(datasets) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[report] Saved: {os.path.join(LATEX_DIR, 'table_runtime.tex')}")


def plot_ecdf(runs: pd.DataFrame):
    """Empirical Cumulative Distribution Function (ECDF) der Test-AUROC über alle Runs.

    Je ein Subplot für MLP und RF.
    """
    runs = runs[~runs["method"].isin(_ACRS_OTHERS)]
    optimizers = sorted(runs["method"].unique())

    splits = [
        ("MLP", runs[runs["model_type"] == "mlp"]),
        ("RF",  runs[runs["model_type"] == "rf"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (title, df) in zip(axes, splits):
        for optimizer in optimizers:
            values = np.sort(df[df["method"] == optimizer]["test_auroc"].values)
            ecdf_y = np.arange(1, len(values) + 1) / len(values)
            ax.step(values, ecdf_y, label=LABELS_MAIN[optimizer],
                    color=PALETTE[optimizer], where="post", linewidth=1.8)

        ax.set_title(title)
        ax.set_xlabel("Test-AUROC")
        ax.set_ylabel("Fraction of Runs")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=runs["test_auroc"].min() - 0.01)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Empirical Cumulative Distribution Function — Test-AUROC", fontsize=14)
    fig.tight_layout()
    _save(fig, os.path.join(PLOTS_DIR, "ecdf_test_auroc.png"))


def plot_acrs_mechanism(filename: str = "acrs_mechanism.pdf"):
    """Schematische Methodik-Abbildung des ACRS-Mechanismus.

    Rein didaktisch, ohne Run-Daten. Tabelle mit Hyperparametern (Zeilen) und
    Runden (Spalten): je Zelle ein Domänen-Balken mit hervorgehobenem Suchbereich
    und einem Punkt für den gewählten Wert. Über die Runden schrumpft der Bereich
    (Eingrenzen), der Punkt markiert den gefundenen Wert. Kategoriale HPs werden
    nicht abgeschnitten und stehen abgesetzt im unteren Block.
    """
    accent = PALETTE["acrs_shrink"]   # ACRS-Akzentfarbe (Vermilion)
    track  = "#E4E4E4"
    grey   = "#8A8A8A"
    R = 3

    # Position eines Werts auf dem Domänen-Balken (linear bzw. logarithmisch)
    def _lin(v, lo, hi): return (v - lo) / (hi - lo)
    def _log(v, lo, hi): return (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))

    # stetige HPs: name, Domänen-Label, lo, hi, log?, [(a, b, v) je Runde], Wertformat
    # der gewählte Wert verfeinert sich über die Runden, der Bereich schrumpft jeweils
    # um den zuletzt gefundenen Wert (der neue Wert liegt innerhalb, aber nicht zentral)
    cont = [
        ("learning_rate", "1e-5..1e-1", 1e-5, 1e-1, True,
         [(1e-5, 1e-1, 5.0e-3), (1.3e-3, 1.9e-2, 8.0e-3), (4e-3, 1.6e-2, 9.0e-3)], "{:.1e}"),
        ("hidden_dim", "32..256", 32, 256, False,
         [(32, 256, 150), (94, 206, 120), (92, 148, 126)], "{:.0f}"),
        ("dropout", "0.0..0.5", 0.0, 0.5, False,
         [(0.0, 0.5, 0.22), (0.10, 0.35, 0.30), (0.24, 0.36, 0.28)], "{:.2f}"),
        ("weight_decay", "1e-6..1e-2", 1e-6, 1e-2, True,
         [(1e-6, 1e-2, 3.0e-4), (3e-5, 3e-3, 1.0e-4), (3e-5, 3.3e-4, 1.3e-4)], "{:.1e}"),
        ("batch_size_exp", "4..7", 4, 7, False,
         [(4, 7, 5), (5, 7, 6), (5, 7, 6)], "{:.0f}"),
    ]
    # kategoriale HPs: name, Auswahl, Index der gewählten (kein Shrink über die Runden)
    cat = [
        ("num_layers", ["1", "2", "3"], 1),
        ("optimizer_name", ["adam", "sgd", "adamw"], 0),
        ("activation", ["relu", "tanh"], 0),
    ]

    n_rows = len(cont) + len(cat)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.set_xlim(0, 1 + R); ax.set_ylim(0, n_rows + 0.8); ax.axis("off")

    def _row_y(i): return n_rows - 0.5 - i

    # Kopfzeile
    ax.text(0.5, n_rows + 0.35, "Hyperparameter", ha="center", va="center", fontweight="bold", fontsize=9)
    for c in range(R):
        ax.text(1 + c + 0.5, n_rows + 0.35, f"Round {c + 1}", ha="center", va="center", fontweight="bold")

    pad = 0.10

    def _draw_track(cx, y, p_a, p_b, p_v, vtxt):
        """Domänen-Balken, hervorgehobener Suchbereich und Punkt für den gewählten Wert."""
        tl, tr = cx + pad, cx + 1 - pad
        w  = tr - tl
        yc = y + 0.05
        ax.add_patch(Rectangle((tl, yc - 0.022), w, 0.044, facecolor=track, edgecolor="none"))
        ax.add_patch(Rectangle((tl + p_a * w, yc - 0.05), (p_b - p_a) * w, 0.10,
                               facecolor=accent, alpha=0.30, edgecolor=accent, lw=0.8))
        ax.plot(tl + p_v * w, yc, "o", color=accent, ms=7, mec="white", mew=0.6, zorder=5)
        ax.text(cx + 0.5, y - 0.20, vtxt, ha="center", va="center", fontsize=8)

    # stetige Zeilen
    for i, (name, dom, lo, hi, is_log, rounds, fmt) in enumerate(cont):
        y = _row_y(i)
        ax.text(0.92, y + 0.05, name, ha="right", va="center", fontsize=9, family="monospace")
        ax.text(0.92, y - 0.20, dom, ha="right", va="center", fontsize=6.5, color=grey)
        f = _log if is_log else _lin
        for c, (a, b, v) in enumerate(rounds):
            _draw_track(1 + c, y, f(a, lo, hi), f(b, lo, hi), f(v, lo, hi), "-> " + fmt.format(v))

    # Trennlinie zwischen stetigem und kategorialem Block
    y_split = _row_y(len(cont)) + 0.5
    ax.plot([0.05, R + 0.95], [y_split, y_split], color="#cccccc", lw=1.0, ls=(0, (4, 3)))

    # kategoriale Zeilen
    for j, (name, choices, chosen) in enumerate(cat):
        y = _row_y(len(cont) + j)
        ax.text(0.92, y + 0.05, name, ha="right", va="center", fontsize=9, family="monospace")
        ax.text(0.92, y - 0.20, "categorical", ha="right", va="center", fontsize=6.5, color=grey)
        for c in range(R):
            cx = 1 + c
            locs = np.linspace(cx + 0.28, cx + 0.72, len(choices))
            for k, x in enumerate(locs):
                if k == chosen:
                    ax.plot(x, y + 0.05, "s", color=accent, ms=9, zorder=5)
                else:
                    ax.plot(x, y + 0.05, "s", mfc="white", mec=grey, ms=8)
            ax.text(cx + 0.5, y - 0.20, "-> " + choices[chosen], ha="center", va="center", fontsize=8)

    # Legende
    ax.text(0.05, -0.15,
            "bar = full domain     highlighted = current search range (shrinks each round)     "
            "dot = selected value     squares = categories (not shrunk)",
            ha="left", va="center", fontsize=7.5, color="#444444")

    fig.suptitle("ACRS: how each hyperparameter is sampled and progressively narrowed",
                 fontsize=12, y=0.98)
    fig.tight_layout()

    # als PDF für die Thesis und als PNG zur Kontrolle speichern
    for path in (os.path.join(GFX_DIR, filename), os.path.join(PLOTS_DIR, "acrs_mechanism.png")):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"[report] Saved: {path}")
    plt.close(fig)


def main():
    runs, trials = _load_data()
    # volles Raster mit Std-Band (Appendix) + gekürzter Ausschnitt ohne Band (Haupttext)
    plot_convergence(trials)
    plot_convergence(trials, MAIN_DATASETS, "convergence_main.png", show_std=False)
    plot_convergence_time(trials)
    plot_convergence_time(trials, MAIN_DATASETS, "convergence_time_main.png", show_std=False)
    generate_final_performance_table(runs, trials)
    generate_acrs_ablation_table(runs, trials)
    generate_efficiency_table(runs)
    generate_latex_tables(runs)
    plot_ecdf(runs)
    # schematische Methodik-Abbildung (ohne Run-Daten)
    plot_acrs_mechanism()


if __name__ == "__main__":
    main()
