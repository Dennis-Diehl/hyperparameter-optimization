"""Konstanten, Suchräume und Seeds für den HPO-Benchmark."""

SEEDS    = [42, 123, 456, 789, 1234, 2345, 3456, 4567, 5678, 6789]
DATASETS = ["breast_cancer", "diabetes", "bank_marketing", "german_credit", "adult_income"]

# Daten-Split: 60/20/20 
TEST_SIZE = 0.2   # 20% für Testen, 80% für Training + Validierung
VAL_SIZE  = 0.25  # von den 80% Trainingsdaten für Validierung 

MAX_EPOCHS              = 100
EARLY_STOPPING_PATIENCE = 10

# ACRS-Budget
ACRS_R     = 3
ACRS_L     = 5
ACRS_ALPHA = 1.0

# Suchraum MLP
MLP_SEARCH_SPACE = {
    "learning_rate":  {"type": "log",        "low": 1e-5,  "high": 1e-1},
    "batch_size_exp": {"type": "int",        "low": 4,    "high": 7},  # nur Exponent, damit Batch Size kontinuierlich ist
    "hidden_dim":     {"type": "int",         "low": 32,    "high": 256},
    "dropout":        {"type": "float",       "low": 0.0,   "high": 0.5},
    "num_layers":     {"type": "categorical", "choices": [1, 2, 3]},
    "optimizer_name": {"type": "categorical", "choices": ["adam", "sgd", "adamw"]},
    "weight_decay":   {"type": "log",         "low": 1e-6,  "high": 1e-2},
    "activation":     {"type": "categorical", "choices": ["relu", "tanh"]},
}

# Suchraum Random Forest
RF_SEARCH_SPACE = {
    "n_estimators":      {"type": "int",         "low": 50,  "high": 500},
    "max_depth":         {"type": "int",         "low": 3,   "high": 30},
    "min_samples_split": {"type": "int",         "low": 2,   "high": 20},
    "max_features":      {"type": "categorical", "choices": ["sqrt", "log2"]},
    "min_samples_leaf":  {"type": "int",         "low": 1,   "high": 10},
    "criterion":         {"type": "categorical", "choices": ["gini", "entropy"]},
    "max_samples":       {"type": "float",       "low": 0.5, "high": 1.0},
}

# Alle Experimente: Kombinationen aus Optimierungsmethode und Modelltyp
EXPERIMENTS = [
    ("bo",          "mlp"),
    ("bo",          "rf"),
    ("acrs",        "mlp"),
    ("acrs",        "rf"),
    ("acrs_normal", "mlp"),
    ("acrs_normal", "rf"),
    ("rs",          "mlp"),
    ("rs",          "rf"),
    ("cmaes",       "mlp"),
    ("cmaes",       "rf"),
]

# Trial-Budget: entspricht dem ACRS-Budget (1 + R·n·L) pro Modell
N_TRIALS_MLP = 1 + ACRS_R * len(MLP_SEARCH_SPACE) * ACRS_L  # 121
N_TRIALS_RF  = 1 + ACRS_R * len(RF_SEARCH_SPACE)  * ACRS_L  # 106
