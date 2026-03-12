"""
Federated Learning — Client
===========================
Each FL client:
  1. Holds a local dataset partition (non-IID by design).
  2. Trains a local model for a fixed number of local epochs.
  3. Exposes a `get_score()` method that returns a scalar fitness value
     (validation loss) used by the server's metaheuristic optimizer.
  4. Accepts a global parameter vector and updates its local model weights.

The "score-based" communication protocol (as in FedPSO / FedGWO / FedALO)
transmits a single scalar score per client instead of full model weight
vectors, dramatically reducing communication cost.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


class FederatedClient:
    """One participating node in the federated system."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        indices: list[int],
        local_epochs: int = 1,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu",
        failure_rate: float = 0.0,
    ):
        """
        Parameters
        ----------
        client_id    : unique integer identifier
        model        : a fresh (or pre-warmed) nn.Module instance
        dataset      : full dataset; the client uses only `indices`
        indices      : sample indices assigned to this client (non-IID partition)
        local_epochs : number of local SGD epochs per FL round
        batch_size   : mini-batch size for local training
        lr           : local learning rate
        device       : 'cpu' or 'cuda'
        failure_rate : probability [0,1] that this client fails to respond (packet drop)
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.failure_rate = failure_rate

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        subset = Subset(dataset, indices)
        self.loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        self._last_val_loss: float = np.inf

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def local_train(self) -> float:
        """
        Run `local_epochs` of training on the client's local data.
        Returns the average training loss (used as fitness score).
        """
        self.model.train()
        total_loss = 0.0
        count = 0
        for _ in range(self.local_epochs):
            for X, y in self.loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(y)
                count += len(y)
        avg_loss = total_loss / (count + 1e-10)
        self._last_val_loss = avg_loss
        return avg_loss

    def get_score(self) -> float:
        """
        Return the fitness score (local training loss) for the optimizer.
        Simulates packet-drop: with probability `failure_rate` return inf.
        """
        if np.random.rand() < self.failure_rate:
            return np.inf            # simulate dropped packet
        return self._last_val_loss

    def get_weights(self) -> np.ndarray:
        """Flatten all model parameters into a 1-D numpy vector."""
        return np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in self.model.parameters()]
        )

    def set_weights(self, weight_vector: np.ndarray):
        """Set model parameters from a 1-D numpy weight vector."""
        ptr = 0
        with torch.no_grad():
            for p in self.model.parameters():
                numel = p.numel()
                p.copy_(
                    torch.tensor(
                        weight_vector[ptr: ptr + numel].reshape(p.shape),
                        dtype=p.dtype,
                    )
                )
                ptr += numel

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Return (loss, accuracy) on an external data loader."""
        self.model.eval()
        total_loss, correct, count = 0.0, 0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                total_loss += self.criterion(logits, y).item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                count += len(y)
        return total_loss / (count + 1e-10), correct / (count + 1e-10)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
