from pathlib import Path

import pytest
import torch

from labdino.model import CharRNN, ModelConfig
from labdino.preprocessing import Vocab, build_dataset, load_names
from labdino.training import evaluate, make_loss, split_train_val

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "name_generator" / "artifacts" / "checkpoint_B.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint_B.pt no existe")
def test_checkpoint_b_reproduces_best_val_loss():
    names = load_names(ROOT / "data" / "dinos.csv")
    vocab = Vocab.from_names(names)
    X, Y, _ = build_dataset(names, vocab)
    _, val = split_train_val(X, Y, val_fraction=0.1, seed=42)

    ckpt = torch.load(CKPT, map_location="cpu")
    model = CharRNN(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])

    val_loss = evaluate(model, make_loss(vocab.pad_id), *val)
    print(f"\nval loss checkpoint_B = {val_loss:.4f} (época {ckpt['best_epoch']})")
    assert ckpt["best_epoch"] == 8
    assert val_loss == pytest.approx(1.5589, abs=2e-3)
