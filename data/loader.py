"""
Data Loading and Non-IID Partitioning
======================================
Provides dataset loaders and a Dirichlet-based non-IID partitioner that
distributes data across FL clients with heterogeneous label distributions,
mimicking real-world federated scenarios as described in the paper.

Non-IID partitioning
--------------------
  Dirichlet(α) distribution over class labels:
    - Small α (e.g. 0.1) → extreme non-IID (each client sees few classes)
    - Large α (e.g. 10)  → near-IID

  This matches the heterogeneous client conditions used in the paper.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

DATA_ROOT = Path("./data")


def load_mnist(batch_size: int = 64) -> tuple[torch.utils.data.Dataset, DataLoader]:
    """Return (train_dataset, test_loader) for MNIST."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(DATA_ROOT, train=True,  download=True, transform=tf)
    test  = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)
    return train, DataLoader(test, batch_size=batch_size, shuffle=False)


def load_cifar10(batch_size: int = 64) -> tuple[torch.utils.data.Dataset, DataLoader]:
    """Return (train_dataset, test_loader) for CIFAR-10."""
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train = datasets.CIFAR10(DATA_ROOT, train=True,  download=True, transform=tf_train)
    test  = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tf_test)
    return train, DataLoader(test, batch_size=batch_size, shuffle=False)


def load_generic_image_folder(
    root: str,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
) -> tuple[torch.utils.data.Dataset, DataLoader]:
    """
    Generic ImageFolder loader for Rice Leaf Disease / WaRP.

    Expects data at `root/` with class sub-directories.
    Returns (train_dataset, test_loader).
    """
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    full = datasets.ImageFolder(root, transform=tf)
    n_val = int(len(full) * val_split)
    n_train = len(full) - n_val
    train, val = random_split(full, [n_train, n_val],
                              generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    return train, test_loader


# ---------------------------------------------------------------------------
# Non-IID partitioner
# ---------------------------------------------------------------------------

def non_iid_partition(
    dataset: torch.utils.data.Dataset,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> list[list[int]]:
    """
    Partition dataset indices among `n_clients` using a Dirichlet distribution
    over class labels (heterogeneous / non-IID split).

    Parameters
    ----------
    dataset  : PyTorch Dataset with .targets attribute (or compatible)
    n_clients: number of federated clients
    alpha    : Dirichlet concentration parameter
                 α → 0  : extremely non-IID
                 α → ∞  : IID
    seed     : random seed for reproducibility

    Returns
    -------
    List of length n_clients; each element is a list of sample indices.
    """
    np.random.seed(seed)

    # Retrieve labels
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "targets"):
        # Handle Subset wrapping
        labels = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    n_classes = int(labels.max()) + 1
    class_indices = [np.where(labels == c)[0].tolist() for c in range(n_classes)]
    for ci in class_indices:
        np.random.shuffle(ci)

    # Dirichlet draw: proportions[k] is how class k is split among clients
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)
        splits = np.split(class_indices[c], proportions[:-1])
        for j, split in enumerate(splits):
            client_indices[j].extend(split.tolist())

    return client_indices


# ---------------------------------------------------------------------------
# Factory: dataset name → (train_dataset, test_loader, client_index_lists)
# ---------------------------------------------------------------------------

def get_dataset_and_partition(
    name: str,
    n_clients: int = 10,
    alpha: float = 0.5,
    batch_size: int = 64,
    data_root: str = None,
) -> tuple[torch.utils.data.Dataset, DataLoader, list[list[int]]]:
    """
    Load dataset and generate non-IID partitioning.

    Parameters
    ----------
    name       : 'mnist', 'cifar10', 'rice', or 'warp'
    n_clients  : number of FL clients
    alpha      : Dirichlet non-IID parameter
    batch_size : test batch size
    data_root  : path for custom image-folder datasets (rice / warp)

    Returns
    -------
    (train_dataset, test_loader, client_index_lists)
    """
    name = name.lower()
    if name == "mnist":
        train, test_loader = load_mnist(batch_size)
    elif name in ("cifar10", "cifar-10"):
        train, test_loader = load_cifar10(batch_size)
    elif name in ("rice", "rice_leaf", "rice_leaf_disease"):
        if data_root is None:
            raise ValueError("data_root must be specified for the Rice Leaf Disease dataset.")
        train, test_loader = load_generic_image_folder(data_root, img_size=224, batch_size=batch_size)
    elif name == "warp":
        if data_root is None:
            raise ValueError("data_root must be specified for the WaRP dataset.")
        train, test_loader = load_generic_image_folder(data_root, img_size=64, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose: mnist, cifar10, rice, warp.")

    client_index_lists = non_iid_partition(train, n_clients, alpha)
    return train, test_loader, client_index_lists
