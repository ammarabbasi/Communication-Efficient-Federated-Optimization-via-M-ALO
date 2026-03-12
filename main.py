"""
main.py — Entry Point for FedM-ALO Experiments
================================================
Reproduces the experiments in:
  "Communication-Efficient Federated Optimization via Entropy-Guided
   Ant Lion Optimizer for Next-Generation Wireless Networks"

Usage
-----
  # Default: MNIST, M-ALO2, 10 clients, C=1.0, 0% packet drop
  python main.py

  # CIFAR-10, 20% packet drop, C=0.5, 30 rounds
  python main.py --dataset cifar10 --failure_rate 0.2 --C 0.5

  # Compare all algorithms on MNIST
  python main.py --dataset mnist --compare_all

  # Full reproducibility sweep (30 runs per algorithm, Wilcoxon test)
  python main.py --dataset mnist --n_runs 30 --wilcoxon

CLI Arguments
-------------
  --dataset        : mnist | cifar10 | rice | warp          (default: mnist)
  --optimizer      : ALO | MALO1 | MALO2                    (default: MALO2)
  --n_clients      : number of FL clients                   (default: 10)
  --rounds         : number of communication rounds         (default: 30)
  --C              : client participation ratio             (default: 1.0)
  --failure_rate   : packet-drop probability [0, 1]         (default: 0.0)
  --alpha          : Dirichlet non-IID parameter            (default: 0.5)
  --n_runs         : independent runs (for Wilcoxon)        (default: 1)
  --compare_all    : run ALO, MALO1, MALO2 and compare      (flag)
  --wilcoxon       : run Wilcoxon test after experiments    (flag)
  --no_plot        : suppress matplotlib plots              (flag)
  --output_dir     : directory for saved results/plots      (default: ./outputs)
  --seed           : global random seed                     (default: 42)
"""

import argparse
import copy
import os
import random
import sys
import numpy as np
import torch

# Ensure project root is importable regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from config import FL_CONFIG, OPTIMIZER_CONFIG, DATASET_CONFIG, OUTPUT_DIR
from data.loader import get_dataset_and_partition
from models.cnn import build_model
from federated.client import FederatedClient
from federated.server import FederatedServer
from utils.metrics import (
    run_wilcoxon_table,
    plot_convergence,
    plot_comm_cost,
    save_results,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Build clients
# ---------------------------------------------------------------------------

def build_clients(
    train_dataset,
    client_index_lists: list[list[int]],
    model_factory,
    local_epochs: int,
    batch_size: int,
    lr: float,
    failure_rate: float,
    device: str,
) -> list[FederatedClient]:
    """Instantiate one FederatedClient per index list."""
    clients = []
    for cid, indices in enumerate(client_index_lists):
        model = model_factory()          # fresh copy for each client
        client = FederatedClient(
            client_id=cid,
            model=model,
            dataset=train_dataset,
            indices=indices,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            failure_rate=failure_rate,
        )
        clients.append(client)
    return clients


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_experiment(
    dataset_name: str,
    optimizer_name: str,
    n_clients: int,
    rounds: int,
    participation_ratio: float,
    failure_rate: float,
    alpha: float,
    seed: int,
    device: str,
    opt_kwargs: dict,
    data_root: str = None,
) -> dict:
    """
    Run one complete federated experiment.

    Returns
    -------
    dict with keys: accuracies, losses, communication_cost, optimizer_history
    """
    set_seed(seed)

    ds_cfg = DATASET_CONFIG.get(dataset_name, {})
    effective_root = data_root or ds_cfg.get("data_root")

    # ── Data ────────────────────────────────────────────────────────────
    train_dataset, test_loader, client_indices = get_dataset_and_partition(
        name=dataset_name,
        n_clients=n_clients,
        alpha=alpha,
        data_root=effective_root,
    )

    # ── Model factory ───────────────────────────────────────────────────
    def model_factory():
        return build_model(dataset_name)

    # ── Clients ─────────────────────────────────────────────────────────
    clients = build_clients(
        train_dataset=train_dataset,
        client_index_lists=client_indices,
        model_factory=model_factory,
        local_epochs=FL_CONFIG["local_epochs"],
        batch_size=FL_CONFIG["batch_size"],
        lr=FL_CONFIG["lr"],
        failure_rate=failure_rate,
        device=device,
    )

    # ── Global model ────────────────────────────────────────────────────
    global_model = model_factory()

    # ── Federated Server ─────────────────────────────────────────────────
    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        optimizer_name=optimizer_name,
        optimizer_kwargs=copy.deepcopy(opt_kwargs),
        participation_ratio=participation_ratio,
        communication_rounds=rounds,
        test_loader=test_loader,
        device=device,
    )

    history = server.train()
    return history


# ---------------------------------------------------------------------------
# Compare multiple optimizers
# ---------------------------------------------------------------------------

def compare_algorithms(
    algorithms: list[str],
    dataset_name: str,
    args,
    device: str,
    opt_kwargs: dict,
) -> dict[str, dict]:
    """Run each algorithm once and return their result dicts."""
    all_results = {}
    for alg in algorithms:
        print(f"\n{'='*60}")
        print(f"  Running: {alg}  |  Dataset: {dataset_name.upper()}")
        print(f"{'='*60}")

        # Map user-facing names to server optimizer keys
        opt_map = {
            "FedAvg":    "ALO",      # FedAvg uses simple averaging; we use ALO as placeholder
            "FedALO":    "ALO",
            "FedM-ALO1": "MALO1",
            "FedM-ALO2": "MALO2",
        }
        opt_name = opt_map.get(alg, alg.replace("Fed", ""))

        result = run_experiment(
            dataset_name=dataset_name,
            optimizer_name=opt_name,
            n_clients=args.n_clients,
            rounds=args.rounds,
            participation_ratio=args.C,
            failure_rate=args.failure_rate,
            alpha=args.alpha,
            seed=args.seed,
            device=device,
            opt_kwargs=copy.deepcopy(opt_kwargs),
        )
        all_results[alg] = result
        final_acc = result["accuracies"][-1] if result["accuracies"] else 0.0
        print(f"  → Final accuracy: {final_acc:.4f}  |  "
              f"Comm. cost: {result['communication_cost']:.4f}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FedM-ALO: Communication-Efficient FL")
    p.add_argument("--dataset",      type=str,   default="mnist",
                   choices=["mnist", "cifar10", "rice", "warp"])
    p.add_argument("--optimizer",    type=str,   default="MALO2",
                   choices=["ALO", "MALO1", "MALO2"])
    p.add_argument("--n_clients",    type=int,   default=FL_CONFIG["n_clients"])
    p.add_argument("--rounds",       type=int,   default=FL_CONFIG["communication_rounds"])
    p.add_argument("--C",            type=float, default=FL_CONFIG["participation_ratio"])
    p.add_argument("--failure_rate", type=float, default=FL_CONFIG["failure_rate"])
    p.add_argument("--alpha",        type=float, default=FL_CONFIG["non_iid_alpha"])
    p.add_argument("--n_runs",       type=int,   default=1)
    p.add_argument("--compare_all",  action="store_true")
    p.add_argument("--wilcoxon",     action="store_true")
    p.add_argument("--no_plot",      action="store_true")
    p.add_argument("--output_dir",   type=str,   default=OUTPUT_DIR)
    p.add_argument("--seed",         type=int,   default=FL_CONFIG["seed"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device}")
    print(f"[INFO] Dataset: {args.dataset.upper()}  |  "
          f"Optimizer: {args.optimizer}  |  "
          f"Clients: {args.n_clients}  |  "
          f"Rounds: {args.rounds}  |  "
          f"C={args.C}  |  "
          f"Failure={args.failure_rate*100:.0f}%\n")

    opt_kwargs = {
        "n":     OPTIMIZER_CONFIG["n"],
        "lam":   OPTIMIZER_CONFIG["lam"],
        "gamma": OPTIMIZER_CONFIG["gamma"],
        "beta":  OPTIMIZER_CONFIG["beta"],
    }

    # ── Mode 1: Compare all algorithms ─────────────────────────────────
    if args.compare_all:
        algorithms = ["FedALO", "FedM-ALO1", "FedM-ALO2"]
        all_results = compare_algorithms(
            algorithms=algorithms,
            dataset_name=args.dataset,
            args=args,
            device=device,
            opt_kwargs=opt_kwargs,
        )

        # Convergence plot
        acc_curves = {alg: r["accuracies"] for alg, r in all_results.items()}
        comm_costs = {alg: r["communication_cost"] for alg, r in all_results.items()}

        if not args.no_plot:
            plot_convergence(
                acc_curves,
                title=f"Convergence — {args.dataset.upper()}",
                save_path=f"{args.output_dir}/convergence_{args.dataset}.png",
            )
            plot_comm_cost(
                comm_costs,
                save_path=f"{args.output_dir}/comm_cost_{args.dataset}.png",
            )

        save_results(
            all_results,
            path=f"{args.output_dir}/results_{args.dataset}_compare.json",
        )

    # ── Mode 2: Wilcoxon multi-run evaluation ───────────────────────────
    elif args.wilcoxon and args.n_runs > 1:
        print(f"[INFO] Running {args.n_runs} independent runs for Wilcoxon test …")
        algorithms = ["FedALO", "FedM-ALO1", "FedM-ALO2"]
        run_records = {alg: {args.dataset: []} for alg in algorithms}

        for run_idx in range(args.n_runs):
            print(f"\n─── Run {run_idx + 1}/{args.n_runs} ───")
            for alg in algorithms:
                opt_map = {"FedALO": "ALO", "FedM-ALO1": "MALO1", "FedM-ALO2": "MALO2"}
                result = run_experiment(
                    dataset_name=args.dataset,
                    optimizer_name=opt_map[alg],
                    n_clients=args.n_clients,
                    rounds=args.rounds,
                    participation_ratio=args.C,
                    failure_rate=args.failure_rate,
                    alpha=args.alpha,
                    seed=args.seed + run_idx,
                    device=device,
                    opt_kwargs=copy.deepcopy(opt_kwargs),
                )
                final_acc = result["accuracies"][-1] if result["accuracies"] else 0.0
                run_records[alg][args.dataset].append(final_acc)

        run_wilcoxon_table(run_records, proposed_key="FedM-ALO2")
        save_results(
            run_records,
            path=f"{args.output_dir}/wilcoxon_{args.dataset}_{args.n_runs}runs.json",
        )

    # ── Mode 3: Single run ───────────────────────────────────────────────
    else:
        result = run_experiment(
            dataset_name=args.dataset,
            optimizer_name=args.optimizer,
            n_clients=args.n_clients,
            rounds=args.rounds,
            participation_ratio=args.C,
            failure_rate=args.failure_rate,
            alpha=args.alpha,
            seed=args.seed,
            device=device,
            opt_kwargs=opt_kwargs,
        )

        print(f"\n[SUMMARY]")
        print(f"  Final Accuracy    : {result['accuracies'][-1]:.4f}")
        print(f"  Final Loss        : {result['losses'][-1]:.4f}")
        print(f"  Communication Cost: {result['communication_cost']:.4f}")

        if not args.no_plot:
            alg_label = f"FedM-{args.optimizer}" if "MALO" in args.optimizer else f"Fed{args.optimizer}"
            plot_convergence(
                {alg_label: result["accuracies"]},
                title=f"{alg_label} — {args.dataset.upper()}",
                save_path=f"{args.output_dir}/convergence_{args.dataset}_{args.optimizer}.png",
            )

        save_results(
            result,
            path=f"{args.output_dir}/result_{args.dataset}_{args.optimizer}.json",
        )


if __name__ == "__main__":
    main()
