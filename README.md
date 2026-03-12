# FedM-ALO: Communication-Efficient Federated Optimization

Implementation of **M-ALO1** and **M-ALO2** — two enhanced Ant Lion Optimizer variants
for communication-efficient Federated Learning — from:

> *"Communication-Efficient Federated Optimization via Entropy-Guided Ant Lion Optimizer
>  for Next-Generation Wireless Networks"*

---

## Project Structure

```
fedmalo/
│
├── main.py                   ← Entry point (CLI)
├── config.py                 ← All hyperparameters (Table 4 in paper)
├── requirements.txt
│
├── optimizers/
│   ├── alo.py                ← Standard ALO (Mirjalili 2015)
│   ├── malo1.py              ← M-ALO1: ALO + Entropy-Guided Shrinking Boundary (EGSB)
│   └── malo2.py              ← M-ALO2: M-ALO1 + Momentum-Based Position Update
│
├── federated/
│   ├── client.py             ← FL client: local training, score reporting, weight exchange
│   └── server.py             ← FL server: optimizer-driven aggregation, round management
│
├── models/
│   └── cnn.py                ← MNISTNet, CIFAR10Net, GeneralCNN + build_model()
│
├── data/
│   └── loader.py             ← Dataset loaders + Dirichlet non-IID partitioner
│
└── utils/
    └── metrics.py            ← Wilcoxon test, convergence plots, comm-cost bar chart
```

---

## Algorithm Summary

| Algorithm  | Key Mechanism                                          |
|------------|--------------------------------------------------------|
| **ALO**    | Random walk, roulette selection, adaptive shrinkage    |
| **M-ALO1** | + Entropy-Guided Shrinking Boundary (EGSB)             |
| **M-ALO2** | + EGSB **and** momentum-based position update          |

### EGSB (M-ALO1)
```
H(t)    = Shannon entropy of normalised fitness distribution
α(t)    = 1 - λ × (1 - H(t) / H_max)
X_min^t = A_sel - α(t) × |A_sel - lb|
X_max^t = A_sel + α(t) × |ub - A_sel|
```

### Momentum Update (M-ALO2)
```
v_i^{t+1} = γ × v_i^t + β × (X_i^temp - X_i^t)
X_i^{t+1} = X_i^t + v_i^{t+1}
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Single run: MNIST with M-ALO2 (default)
python main.py

# CIFAR-10 with 20% packet drop, C=0.5
python main.py --dataset cifar10 --failure_rate 0.2 --C 0.5

# Compare FedALO vs FedM-ALO1 vs FedM-ALO2 on MNIST
python main.py --dataset mnist --compare_all

# 30-run Wilcoxon statistical validation
python main.py --dataset mnist --n_runs 30 --wilcoxon

# Use M-ALO1 only, no plots
python main.py --optimizer MALO1 --no_plot
```

---

## CLI Reference

| Flag            | Default | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `--dataset`     | mnist   | mnist / cifar10 / rice / warp                    |
| `--optimizer`   | MALO2   | ALO / MALO1 / MALO2                              |
| `--n_clients`   | 10      | Number of FL clients                             |
| `--rounds`      | 30      | Communication rounds                             |
| `--C`           | 1.0     | Participation ratio (0.1 / 0.2 / 0.5 / 1.0)    |
| `--failure_rate`| 0.0     | Packet-drop rate (0 / 0.1 / 0.2 / 0.5)          |
| `--alpha`       | 0.5     | Dirichlet α for non-IID (smaller = more skewed)  |
| `--n_runs`      | 1       | Independent runs (use ≥ 30 for Wilcoxon)         |
| `--compare_all` | —       | Run and compare all three FL-ALO variants        |
| `--wilcoxon`    | —       | Run Wilcoxon test (requires `--n_runs ≥ 2`)      |
| `--no_plot`     | —       | Suppress matplotlib output                       |
| `--output_dir`  | outputs | Directory for JSON + PNG outputs                 |
| `--seed`        | 42      | Global random seed                               |

---

## Key Hyperparameters (config.py)

| Parameter | Symbol | Value | Description                             |
|-----------|--------|-------|-----------------------------------------|
| `lam`     | λ      | 0.5   | Entropy sensitivity for EGSB            |
| `gamma`   | γ      | 0.9   | Momentum coefficient (M-ALO2)           |
| `beta`    | β      | 0.1   | Displacement weight (M-ALO2)            |
| `n`       | —      | 10    | Population size (ants / ant lions)      |

---

## Custom Datasets (Rice / WaRP)

Set the dataset path in `config.py`:

```python
DATASET_CONFIG["rice"]["data_root"] = "/path/to/rice_leaf_disease"
DATASET_CONFIG["warp"]["data_root"] = "/path/to/warp_c"
```

Then run:

```bash
python main.py --dataset rice --compare_all
```

Expected folder structure (ImageFolder format):
```
rice_leaf_disease/
  blast/      img1.jpg ...
  blight/     img1.jpg ...
  brown_spot/ ...
  leaf_smut/  ...
  tungro/     ...
```

---

## Outputs

All results are written to `./outputs/` (or `--output_dir`):

- `result_<dataset>_<optimizer>.json` — single-run history
- `results_<dataset>_compare.json`   — multi-algorithm comparison
- `convergence_<dataset>.png`        — accuracy vs round plot
- `comm_cost_<dataset>.png`          — normalised communication cost bar chart
- `wilcoxon_<dataset>_<N>runs.json`  — Wilcoxon test run data
