"""Muestreo autoregresivo con temperatura, top-k y top-p, y métricas para compararlo."""

from dataclasses import asdict, dataclass

import torch

from labdino.preprocessing import Vocab, normalize_name


@dataclass(frozen=True)
class SamplingConfig:
    name: str
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None


def next_token_probs(
    logits: torch.Tensor,
    banned_ids: list[int] = (),
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """logits (N, V) -> probabilidades (N, V) tras enmascarar, escalar por T y filtrar con top-k y top-p.

    - `banned_ids` (p. ej. <PAD> y <SOS>) reciben probabilidad 0.
    - k mayor que el vocabulario equivale a no filtrar.
    - top-p conserva siempre al menos el token más probable.
    """
    logits = logits.clone()
    if len(banned_ids):
        logits[:, list(banned_ids)] = float("-inf")
    logits = logits / temperature

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = torch.softmax(logits, dim=-1)

    if top_p is not None:
        sorted_p, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        # Se descarta un token si la masa acumulada ANTES de él ya alcanza p (el primero siempre se queda)
        drop_sorted = (sorted_p.cumsum(-1) - sorted_p) >= top_p
        drop_sorted[:, 0] = False  # con p <= 0 también se conserva el más probable
        drop = torch.zeros_like(drop_sorted).scatter(-1, sorted_idx, drop_sorted)
        probs = probs.masked_fill(drop, 0.0)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs


@torch.no_grad()
def generate_names(
    model: torch.nn.Module,
    vocab: Vocab,
    max_len: int,
    n: int,
    cfg: SamplingConfig,
    generator: torch.Generator,
) -> list[str]:
    """Genera n nombres en paralelo: cada carácter producido se usa como entrada del siguiente paso.

    Parte de <SOS>; un nombre termina en <EOS> o al llegar a `max_len` (= T) caracteres.
    """
    model.eval()
    banned = [vocab.pad_id, vocab.sos_id]
    x = torch.full((n, 1), vocab.sos_id, dtype=torch.long)
    hidden = None
    out = torch.full((n, max_len), vocab.pad_id, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool)

    for t in range(max_len):
        logits, hidden = model(x, hidden)
        probs = next_token_probs(logits[:, -1], banned, cfg.temperature, cfg.top_k, cfg.top_p)
        nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, vocab.pad_id), nxt)
        out[:, t] = nxt
        done |= nxt == vocab.eos_id
        if done.all():
            break
        x = nxt.unsqueeze(1)
    return [vocab.decode(row.tolist()) for row in out]


def is_novel(name: str, known: set[str]) -> bool:
    """True si el nombre (con la misma normalización) no está en el conjunto conocido."""
    return normalize_name(name) not in known


def evaluate_strategy(names: list[str], known: set[str], min_len: int = 3) -> dict:
    """Métricas de una tanda de muestras: copias, repetidos, longitud media y nombres muy cortos."""
    n = len(names)
    return {
        "n_samples": n,
        "copy_rate": sum(not is_novel(s, known) for s in names) / n,
        "repeat_rate": 1 - len(set(names)) / n,
        "mean_length": sum(len(s) for s in names) / n,
        "short_rate": sum(len(s) < min_len for s in names) / n,
    }


def pick_novel_names(
    model: torch.nn.Module,
    vocab: Vocab,
    max_len: int,
    cfg: SamplingConfig,
    known: set[str],
    count: int = 10,
    min_len: int = 3,
    seed: int = 123,
    batch: int = 200,
) -> list[str]:
    """`count` nombres distintos entre sí, no presentes en `known` y de al menos `min_len` caracteres."""
    g = torch.Generator().manual_seed(seed)
    picked: dict[str, None] = {}
    for _ in range(100):
        for s in generate_names(model, vocab, max_len, batch, cfg, g):
            if len(s) >= min_len and is_novel(s, known):
                picked.setdefault(s)
            if len(picked) == count:
                return list(picked)
    raise RuntimeError("No se lograron generar suficientes nombres novedosos")


def config_to_dict(cfg: SamplingConfig) -> dict:
    return asdict(cfg)
