"""
Federated Learning — Server
============================
The server orchestrates FL rounds:
  1. Selects a subset of clients (participation ratio C).
  2. Tells each client to perform local training.
  3. Collects scalar fitness scores (NOT full weights) from clients.
  4. Uses the chosen metaheuristic optimizer (ALO / M-ALO1 / M-ALO2) whose
     fitness function is defined over the space of client participation weights
     to select and aggregate model parameters from high-scoring clients.
  5. Broadcasts the aggregated global model back to all clients.

Communication-cost accounting
------------------------------
  FedAvg   : n_selected * W   (W = number of model parameters)
  FedM-ALO : n_selected * 1   (score only, dramatically lower)
  We track normalised cost = total_scores_sent / (max_rounds * n_clients * W)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .client import FederatedClient
from optimizers.alo import ALO
from optimizers.malo1 import MALO1
from optimizers.malo2 import MALO2


class FederatedServer:
    """Central aggregator for federated metaheuristic optimization."""

    OPTIMIZER_MAP = {
        "ALO": ALO,
        "MALO1": MALO1,
        "MALO2": MALO2,
    }

    def __init__(
        self,
        global_model: nn.Module,
        clients: list[FederatedClient],
        optimizer_name: str = "MALO2",
        optimizer_kwargs: dict = None,
        participation_ratio: float = 1.0,
        communication_rounds: int = 30,
        test_loader: DataLoader = None,
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        global_model        : the shared model architecture (weights are broadcast)
        clients             : list of FederatedClient instances
        optimizer_name      : 'ALO', 'MALO1', or 'MALO2'
        optimizer_kwargs    : extra kwargs forwarded to the optimizer constructor
        participation_ratio : C — fraction of clients selected per round
        communication_rounds: T — total FL rounds
        test_loader         : DataLoader for global evaluation
        device              : 'cpu' or 'cuda'
        """
        self.global_model = global_model.to(device)
        self.clients = clients
        self.optimizer_name = optimizer_name.upper()
        self.opt_kwargs = optimizer_kwargs or {}
        self.C = participation_ratio
        self.T = communication_rounds
        self.test_loader = test_loader
        self.device = device

        # Metrics
        self.round_accuracies: list[float] = []
        self.round_losses: list[float] = []
        self.communication_cost: float = 0.0   # cumulative normalised cost

        # Number of model parameters (for cost accounting)
        self._W = sum(p.numel() for p in global_model.parameters())

        # Build optimizer: it searches over a d-dimensional score space
        # (one dimension per client); each candidate is a weight vector
        # that the aggregation step interprets as importance weights.
        n_clients = len(clients)
        OptClass = self.OPTIMIZER_MAP.get(self.optimizer_name)
        if OptClass is None:
            raise ValueError(
                f"Unknown optimizer '{optimizer_name}'. "
                f"Choose from {list(self.OPTIMIZER_MAP.keys())}."
            )

        self.optimizer = OptClass(
            n=self.opt_kwargs.pop("n", 10),
            d=n_clients,
            max_iter=communication_rounds,
            lb=0.0,
            ub=1.0,
            fitness_fn=None,       # set dynamically each round
            **self.opt_kwargs,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """
        Run all FL communication rounds.
        Returns a dictionary of training history.
        """
        self._broadcast_global_model()
        self.optimizer.initialize()

        for rnd in range(1, self.T + 1):
            print(f"\n[Round {rnd:>3}/{self.T}]", end="  ")

            # 1. Select clients for this round
            selected = self._select_clients()
            print(f"Clients selected: {[c.client_id for c in selected]}", end="  ")

            # 2. Local training on selected clients
            scores = {}
            for client in selected:
                client.local_train()
                scores[client.client_id] = client.get_score()

            # 3. Define fitness function for the optimizer this round
            def make_fitness(sel_clients, score_dict):
                def fitness_fn(weight_vec: np.ndarray) -> float:
                    """
                    Weighted-average the client scores.
                    weight_vec[i] ∈ [0,1] is the importance of client i.
                    Returns the weighted loss (lower is better).
                    """
                    total, total_w = 0.0, 0.0
                    for c in sel_clients:
                        w = float(weight_vec[c.client_id])
                        s = score_dict.get(c.client_id, np.inf)
                        if s < np.inf:
                            total += w * s
                            total_w += w
                    return total / (total_w + 1e-10)
                return fitness_fn

            self.optimizer.fitness_fn = make_fitness(selected, scores)

            # 4. Optimizer step
            best_weights, best_fitness = self.optimizer.step(rnd)

            # 5. Aggregate global model using the importance weights
            self._aggregate(selected, best_weights, scores)

            # 6. Track communication cost
            #    Each selected client sends only 1 scalar (score-based protocol)
            n_available = sum(1 for c in selected if scores[c.client_id] < np.inf)
            self.communication_cost += n_available / (len(self.clients) * self._W)

            # 7. Evaluate global model
            loss, acc = self._evaluate_global()
            self.round_losses.append(loss)
            self.round_accuracies.append(acc)
            print(f"Loss: {loss:.4f}  Acc: {acc:.4f}  Best-fitness: {best_fitness:.4f}")

            # 8. Broadcast updated global model
            self._broadcast_global_model()

        return {
            "accuracies": self.round_accuracies,
            "losses": self.round_losses,
            "communication_cost": self.communication_cost,
            "optimizer_history": self.optimizer.history,
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        selected: list[FederatedClient],
        importance: np.ndarray,
        scores: dict,
    ):
        """
        Weighted-average aggregation of client model parameters.
        Only clients whose packet was received (score < inf) contribute.
        """
        weighted_params = None
        total_weight = 0.0

        for client in selected:
            cid = client.client_id
            s = scores.get(cid, np.inf)
            if s == np.inf:
                continue                       # packet dropped, skip

            w = max(float(importance[cid]), 1e-6)
            # Invert score: better (lower loss) → higher aggregation weight
            eff_weight = w / (s + 1e-10)
            params = client.get_weights()

            if weighted_params is None:
                weighted_params = eff_weight * params
            else:
                weighted_params += eff_weight * params
            total_weight += eff_weight

        if weighted_params is None or total_weight == 0:
            return   # no updates received this round

        agg_params = weighted_params / total_weight
        self._set_global_weights(agg_params)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_clients(self) -> list[FederatedClient]:
        k = max(1, int(self.C * len(self.clients)))
        return list(np.random.choice(self.clients, size=k, replace=False))

    def _broadcast_global_model(self):
        """Copy global model weights to every client."""
        global_weights = self._get_global_weights()
        for client in self.clients:
            client.set_weights(global_weights)

    def _get_global_weights(self) -> np.ndarray:
        return np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in self.global_model.parameters()]
        )

    def _set_global_weights(self, weight_vector: np.ndarray):
        ptr = 0
        with torch.no_grad():
            for p in self.global_model.parameters():
                numel = p.numel()
                p.copy_(
                    torch.tensor(
                        weight_vector[ptr: ptr + numel].reshape(p.shape),
                        dtype=p.dtype,
                    )
                )
                ptr += numel

    def _evaluate_global(self) -> tuple[float, float]:
        """Evaluate the global model on the test loader."""
        if self.test_loader is None:
            return 0.0, 0.0
        criterion = nn.CrossEntropyLoss()
        self.global_model.eval()
        total_loss, correct, count = 0.0, 0, 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.global_model(X)
                total_loss += criterion(logits, y).item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                count += len(y)
        return total_loss / (count + 1e-10), correct / (count + 1e-10)
