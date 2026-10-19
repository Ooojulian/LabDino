"""Modelo de lenguaje a nivel de carácter: Embedding -> recurrente (RNN/LSTM/GRU) -> Linear."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


@dataclass
class ModelConfig:
    vocab_size: int
    cell: str = "gru"  # "rnn" | "lstm" | "gru"
    embedding_dim: int = 32
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.0  # solo se aplica entre capas recurrentes (num_layers > 1)
    pad_id: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class CharRNN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_id)
        self.rnn = CELLS[config.cell](
            config.embedding_dim,
            config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """x: (N, T) de ids -> (logits (N, T, V), estado oculto). `hidden` permite generar paso a paso."""
        out, hidden = self.rnn(self.embedding(x), hidden)
        return self.out(out), hidden
