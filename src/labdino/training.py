"""Pérdida y partición de datos para el entrenamiento."""

import copy
from dataclasses import dataclass, field

import torch
from torch import nn


def make_loss(pad_id: int) -> nn.CrossEntropyLoss:
    """Entropía cruzada que excluye <PAD> del cálculo."""
    return nn.CrossEntropyLoss(ignore_index=pad_id)


def sequence_loss(loss_fn: nn.Module, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """logits (N, T, V) y objetivos (N, T) -> pérdida escalar."""
    return loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def split_train_val(
    X: torch.Tensor, Y: torch.Tensor, val_fraction: float = 0.1, seed: int = 42
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Partición aleatoria reproducible en (X_train, Y_train), (X_val, Y_val)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.size(0), generator=g)
    n_val = max(1, int(X.size(0) * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (X[train_idx], Y[train_idx]), (X[val_idx], Y[val_idx])


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 3e-3
    max_epochs: int = 100
    patience: int = 10  # épocas sin mejorar la pérdida de validación antes de parar
    seed: int = 42


@dataclass
class TrainResult:
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int  # 1-indexada
    best_val_loss: float
    epochs_run: int
    best_state: dict = field(repr=False)  # state_dict del mejor checkpoint (no el último)


@torch.no_grad()
def evaluate(model: nn.Module, loss_fn: nn.Module, X: torch.Tensor, Y: torch.Tensor, batch_size: int = 256) -> float:
    """Pérdida media por token (sin <PAD>) en modo eval y sin gradientes."""
    model.eval()
    total, count = 0.0, 0
    pad_id = loss_fn.ignore_index
    for i in range(0, X.size(0), batch_size):
        x, y = X[i : i + batch_size], Y[i : i + batch_size]
        logits, _ = model(x)
        n_tokens = (y != pad_id).sum().item()
        total += sequence_loss(loss_fn, logits, y).item() * n_tokens
        count += n_tokens
    return total / count


def train_model(
    model: nn.Module,
    train: tuple[torch.Tensor, torch.Tensor],
    val: tuple[torch.Tensor, torch.Tensor],
    pad_id: int,
    cfg: TrainConfig,
) -> TrainResult:
    """Adam + early stopping por paciencia sobre la pérdida de validación; conserva el mejor checkpoint."""
    loss_fn = make_loss(pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    X, Y = train
    g = torch.Generator().manual_seed(cfg.seed)  # mismo orden de batches para cada configuración

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val, best_epoch, best_state = float("inf"), 0, copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(X.size(0), generator=g)
        total, count = 0.0, 0
        for i in range(0, X.size(0), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            x, y = X[idx], Y[idx]
            logits, _ = model(x)
            loss = sequence_loss(loss_fn, logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_tokens = (y != pad_id).sum().item()
            total += loss.item() * n_tokens
            count += n_tokens
        train_losses.append(total / count)
        val_losses.append(evaluate(model, loss_fn, *val))

        if val_losses[-1] < best_val:
            best_val, best_epoch = val_losses[-1], epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break

    return TrainResult(train_losses, val_losses, best_epoch, best_val, len(train_losses), best_state)
