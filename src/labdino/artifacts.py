"""Guardado y carga de los artefactos del generador de nombres (sin reentrenar).

Archivos en el directorio de artefactos:
    model.pt          pesos (state_dict) de la mejor época de la configuración ganadora
    vocab.json        lista de tokens; el índice es el id
    config.json       arquitectura, T, ids de tokens especiales, muestreo por defecto, criterio de selección
    train_names.txt   todos los nombres normalizados (train+val), para rechazar copias
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from labdino.model import CharRNN, ModelConfig
from labdino.preprocessing import Vocab
from labdino.sampling import SamplingConfig, generate_names, is_novel

MODEL_FILE, VOCAB_FILE, CONFIG_FILE, NAMES_FILE = "model.pt", "vocab.json", "config.json", "train_names.txt"
DEFAULT_SAMPLING = {"temperature": 1.0, "top_k": None, "top_p": 0.9}


@dataclass
class Artifacts:
    model: CharRNN
    vocab: Vocab
    max_len: int  # T
    config: dict
    known_names: set[str]
    sampling: SamplingConfig


def save_artifacts(
    out_dir: Path,
    state_dict: dict,
    model_config: ModelConfig,
    vocab: Vocab,
    max_len: int,
    names: list[str],
    selection: dict,
    sampling: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, out_dir / MODEL_FILE)
    (out_dir / VOCAB_FILE).write_text(json.dumps(vocab.itos, ensure_ascii=False, indent=1), encoding="utf-8")
    config = {
        "architecture": model_config.cell.upper(),
        "num_layers": model_config.num_layers,
        "hidden_size": model_config.hidden_size,
        "dropout": model_config.dropout,
        "model": model_config.to_dict(),
        "T": max_len,
        "special_tokens": {"pad": vocab.pad_id, "sos": vocab.sos_id, "eos": vocab.eos_id},
        "default_sampling": sampling or DEFAULT_SAMPLING,
        "selection": selection,
    }
    (out_dir / CONFIG_FILE).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / NAMES_FILE).write_text("\n".join(names) + "\n", encoding="utf-8")


def load_artifacts(directory: Path) -> Artifacts:
    """Carga todo SOLO desde `directory`; los pesos con weights_only=True y en CPU."""
    directory = Path(directory)
    config = json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8"))
    vocab = Vocab(json.loads((directory / VOCAB_FILE).read_text(encoding="utf-8")))
    tokens = config["special_tokens"]
    assert (vocab.pad_id, vocab.sos_id, vocab.eos_id) == (tokens["pad"], tokens["sos"], tokens["eos"])

    model = CharRNN(ModelConfig(**config["model"]))
    model.load_state_dict(torch.load(directory / MODEL_FILE, map_location="cpu", weights_only=True))
    model.eval()

    known = {line for line in (directory / NAMES_FILE).read_text(encoding="utf-8").splitlines() if line}
    sampling = SamplingConfig("default", **config["default_sampling"])
    return Artifacts(model, vocab, config["T"], config, known, sampling)


def generate_novel_name(
    art: Artifacts, generator: torch.Generator | None = None, min_len: int = 3, batch: int = 32, max_tries: int = 100
) -> str:
    """Un nombre que no está en el conjunto de entrenamiento, con el muestreo por defecto.

    Sin `generator` la semilla es aleatoria (torch.Generator() solo tiene una semilla fija por defecto);
    pasa uno con semilla para reproducir el resultado.
    """
    if generator is None:
        generator = torch.Generator()
        generator.seed()  # semilla no determinista
    for _ in range(max_tries):
        for name in generate_names(art.model, art.vocab, art.max_len, batch, art.sampling, generator):
            if len(name) >= min_len and is_novel(name, art.known_names):
                return name
    raise RuntimeError("No se logró generar un nombre novedoso")
