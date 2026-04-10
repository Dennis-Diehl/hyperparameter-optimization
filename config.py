"""Konstanten, Suchräume und Seeds für den HPO-Benchmark."""

from ray import tune

SEEDS    = [42, 123, 456]
DATASETS = ["breast_cancer", "diabetes", "bank_marketing"]

# Daten-Split: 60/20/20 
TEST_SIZE = 0.2   # 20% für Testen, 80% für Training + Validierung
VAL_SIZE  = 0.25  # von den 80% Trainingsdaten für Validierung 

MAX_EPOCHS              = 100
EARLY_STOPPING_PATIENCE = 10

N_TRIALS = 110  # Trials pro BO-Run 

# ACRS-Budget
ACRS_R     = 3
ACRS_L     = 5
ACRS_ALPHA = 0.5  # erstmal nicht genutzt

# Suchraum MLP für BO (Ray-Tune-Format)
MLP_SEARCH_SPACE = {
    "learning_rate":  tune.loguniform(1e-5, 1e-1),
    "batch_size":     tune.choice([16, 32, 64, 128]),
    "hidden_dim":     tune.randint(32, 257),
    "dropout":        tune.uniform(0.0, 0.5),
    "num_layers":     tune.choice([1, 2, 3]),
    "optimizer_name": tune.choice(["adam", "sgd", "adamw"]),
    "weight_decay":   tune.loguniform(1e-6, 1e-2),
    "activation":     tune.choice(["relu", "tanh"]),
}

# Suchraum MLP für ACRS
MLP_SEARCH_SPACE_ACRS = {
    "learning_rate":  {"type": "log",        "low": 1e-5,  "high": 1e-1},
    "batch_size":     {"type": "categorical", "choices": [16, 32, 64, 128]},
    "hidden_dim":     {"type": "int",         "low": 32,    "high": 256},
    "dropout":        {"type": "float",       "low": 0.0,   "high": 0.5},
    "num_layers":     {"type": "categorical", "choices": [1, 2, 3]},
    "optimizer_name": {"type": "categorical", "choices": ["adam", "sgd", "adamw"]},
    "weight_decay":   {"type": "log",         "low": 1e-6,  "high": 1e-2},
    "activation":     {"type": "categorical", "choices": ["relu", "tanh"]},
}

# Suchraum Random Forest für BO (Ray-Tune-Format)
RF_SEARCH_SPACE = {
    "n_estimators":      tune.randint(50, 501),
    "max_depth":         tune.randint(3, 31),
    "min_samples_split": tune.randint(2, 21),
    "max_features":      tune.choice(["sqrt", "log2"]),
    "min_samples_leaf":  tune.randint(1, 11),
    "criterion":         tune.choice(["gini", "entropy"]),
    "max_samples":       tune.uniform(0.5, 1.0),
}

# Suchraum Random Forest für ACRS
RF_SEARCH_SPACE_ACRS = {
    "n_estimators":      {"type": "int",         "low": 50,  "high": 500},
    "max_depth":         {"type": "int",         "low": 3,   "high": 30},
    "min_samples_split": {"type": "int",         "low": 2,   "high": 20},
    "max_features":      {"type": "categorical", "choices": ["sqrt", "log2"]},
    "min_samples_leaf":  {"type": "int",         "low": 1,   "high": 10},
    "criterion":         {"type": "categorical", "choices": ["gini", "entropy"]},
    "max_samples":       {"type": "float",       "low": 0.5, "high": 1.0},
}
