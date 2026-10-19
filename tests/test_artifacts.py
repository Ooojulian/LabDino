import shutil
from pathlib import Path

import pytest
import torch

from labdino.artifacts import generate_novel_name, load_artifacts
from labdino.preprocessing import build_dataset
from labdino.training import evaluate, make_loss, split_train_val

ARTIFACTS = Path(__file__).resolve().parents[1] / "name_generator" / "artifacts"


@pytest.fixture()
def clean_art(tmp_path):
    """Copia SOLO artifacts/ a un directorio limpio: sin data/, evidence/ ni checkpoints."""
    shutil.copytree(ARTIFACTS, tmp_path / "artifacts")
    return load_artifacts(tmp_path / "artifacts")


def test_config_contents(clean_art):
    cfg = clean_art.config
    assert (cfg["architecture"], cfg["num_layers"], cfg["hidden_size"], cfg["dropout"]) == ("GRU", 2, 256, 0.2)
    assert cfg["T"] == clean_art.max_len == 27
    assert cfg["special_tokens"] == {"pad": 0, "sos": 1, "eos": 2}
    assert cfg["default_sampling"] == {"temperature": 1.0, "top_k": None, "top_p": 0.9}
    assert cfg["selection"]["criterion"] == "menor pérdida de validación"
    assert len(clean_art.known_names) == 1524


def test_generates_novel_name_without_retraining(clean_art):
    name = generate_novel_name(clean_art, torch.Generator().manual_seed(0))
    print(f"\nnombre generado desde artefactos: {name.capitalize()}")
    assert len(name) >= 3 and name not in clean_art.known_names


def test_default_seed_is_random_but_explicit_seed_reproduces(clean_art):
    assert generate_novel_name(clean_art, torch.Generator().manual_seed(1)) == generate_novel_name(
        clean_art, torch.Generator().manual_seed(1)
    )
    names = {generate_novel_name(clean_art) for _ in range(5)}  # sin generador: semilla aleatoria
    assert len(names) > 1


def test_loaded_model_reproduces_val_loss(clean_art):
    # train_names.txt conserva el orden del dataset, así que se reconstruye el mismo split (semilla 42, 10 %)
    names = [n for n in (ARTIFACTS / "train_names.txt").read_text().splitlines() if n]
    X, Y, T = build_dataset(names, clean_art.vocab)
    assert T == clean_art.max_len
    _, val = split_train_val(X, Y, val_fraction=0.1, seed=42)
    val_loss = evaluate(clean_art.model, make_loss(clean_art.vocab.pad_id), *val)
    print(f"\nval loss del modelo cargado = {val_loss:.4f}")
    assert val_loss == pytest.approx(1.5589, abs=2e-3)
