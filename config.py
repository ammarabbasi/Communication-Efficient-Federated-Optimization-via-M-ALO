"""
config.py — Centralised Experiment Configuration
=================================================
All hyperparameters from Table 4 of the paper are defined here.
Override any value by editing this file or passing kwargs to the server.
"""

# ── Federated Learning Setup ──────────────────────────────────────────────
FL_CONFIG = {
    "n_clients": 10,                # Total participating clients
    "communication_rounds": 30,     # Maximum FL rounds  (= optimizer max_iter)
    "local_epochs": 1,              # Local SGD epochs per round
    "batch_size": 32,               # Local mini-batch size
    "lr": 1e-3,                     # Local learning rate (Adam)
    "participation_ratio": 1.0,     # C — client fraction per round (0.1 / 0.2 / 0.5 / 1.0)
    "non_iid_alpha": 0.5,           # Dirichlet α for label heterogeneity
    "failure_rate": 0.0,            # Packet-drop probability (0 / 0.1 / 0.2 / 0.5)
    "n_independent_runs": 30,       # Runs for Wilcoxon test
    "seed": 42,
}

# ── Optimizer Hyperparameters (shared) ───────────────────────────────────
OPTIMIZER_CONFIG = {
    "n": 10,           # Population size (ants / ant lions)
    # M-ALO1 specific
    "lam": 0.5,        # λ — entropy sensitivity for boundary shrinkage
    # M-ALO2 specific (also inherits lam)
    "gamma": 0.9,      # γ — momentum coefficient
    "beta": 0.1,       # β — displacement weight
}

# ── Available Algorithms ─────────────────────────────────────────────────
ALGORITHMS = ["FedAvg", "FedALO", "FedM-ALO1", "FedM-ALO2"]
# Extend with: "FedPSO", "FedSCA", "FedGWO" if those optimizers are available

# ── Dataset Configuration ─────────────────────────────────────────────────
DATASET_CONFIG = {
    "mnist": {
        "name": "mnist",
        "data_root": None,          # auto-downloaded
        "n_classes": 10,
    },
    "cifar10": {
        "name": "cifar10",
        "data_root": None,          # auto-downloaded
        "n_classes": 10,
    },
    "rice": {
        "name": "rice",
        "data_root": "./data/rice_leaf_disease",   # set your local path
        "n_classes": 5,
    },
    "warp": {
        "name": "warp",
        "data_root": "./data/warp_c",              # set your local path
        "n_classes": 28,
    },
}

# ── Output Paths ─────────────────────────────────────────────────────────
OUTPUT_DIR = "./outputs"
