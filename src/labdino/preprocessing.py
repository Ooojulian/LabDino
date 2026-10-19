"""Preprocesamiento de nombres para el modelo de caracteres.

Orden de la guía: normalizar -> vocabulario -> tokens especiales -> T -> padding -> (X, Y).
"""

from dataclasses import dataclass
from pathlib import Path

import torch

PAD, SOS, EOS = "<PAD>", "<SOS>", "<EOS>"
SPECIAL_TOKENS = [PAD, SOS, EOS]  # <PAD> siempre tiene id 0


def normalize_name(name: str) -> str:
    """Mismo criterio de escritura para todos: sin espacios en los bordes y en minúsculas."""
    return name.strip().lower()


def load_names(path: str | Path) -> list[str]:
    """Lee un nombre por línea, normaliza y elimina vacíos y duplicados (conserva el orden)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    names = (normalize_name(line) for line in lines)
    return list(dict.fromkeys(n for n in names if n))


@dataclass
class Vocab:
    """Relación carácter <-> identificador numérico, incluidos los tokens especiales."""

    itos: list[str]

    def __post_init__(self) -> None:
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @classmethod
    def from_names(cls, names: list[str]) -> "Vocab":
        chars = sorted({c for n in names for c in n})
        return cls(SPECIAL_TOKENS + chars)

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def sos_id(self) -> int:
        return self.stoi[SOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def encode(self, name: str) -> list[int]:
        """[SOS, c1, ..., cn, EOS] sin padding."""
        return [self.sos_id, *(self.stoi[c] for c in name), self.eos_id]

    def decode(self, ids: list[int]) -> str:
        """Ids -> texto, omitiendo tokens especiales."""
        return "".join(self.itos[i] for i in ids if self.itos[i] not in SPECIAL_TOKENS)


def build_dataset(names: list[str], vocab: Vocab) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Devuelve (X, Y, T) con X=[x0..x(T-1)], Y=[x1..xT]; ambos de forma (N, T).

    Cada secuencia [x0..xT] es [SOS]+nombre+[EOS] rellenada con <PAD>; T es su longitud máxima - 1.
    """
    seqs = [vocab.encode(n) for n in names]
    T = max(len(s) for s in seqs) - 1
    padded = torch.full((len(seqs), T + 1), vocab.pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s)
    return padded[:, :-1], padded[:, 1:], T
