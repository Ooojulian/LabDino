import math
from pathlib import Path

import pytest
import torch

from labdino.model import CharRNN, ModelConfig
from labdino.preprocessing import Vocab, build_dataset, load_names
from labdino.seed import set_seed
from labdino.training import make_loss, sequence_loss, split_train_val

DATA = Path(__file__).resolve().parents[1] / "data" / "dinos.csv"


@pytest.fixture(scope="module")
def data():
    names = load_names(DATA)
    vocab = Vocab.from_names(names)
    X, Y, T = build_dataset(names, vocab)
    return vocab, X, Y, T


@pytest.mark.parametrize("cell", ["rnn", "lstm", "gru"])
def test_forward_backward_and_initial_loss(data, cell):
    vocab, X, Y, T = data
    set_seed(0)
    model = CharRNN(ModelConfig(len(vocab), cell=cell, num_layers=2, dropout=0.1, pad_id=vocab.pad_id))
    loss_fn = make_loss(vocab.pad_id)

    logits, _ = model(X[:64])
    assert logits.shape == (64, T, len(vocab))
    loss = sequence_loss(loss_fn, logits, Y[:64])
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())

    expected = math.log(len(vocab))
    print(f"\n[{cell}] pérdida inicial={loss.item():.3f}  ln|V|={expected:.3f}")
    assert abs(loss.item() - expected) < 0.5


def test_loss_ignores_pad(data):
    vocab, _, _, _ = data
    loss_fn = make_loss(vocab.pad_id)
    logits = torch.randn(1, 3, len(vocab))
    y = torch.tensor([[5, 6, vocab.pad_id]])
    # Cambiar los logits de la posición con <PAD> no altera la pérdida
    other = logits.clone()
    other[0, 2] += 100
    assert torch.isclose(sequence_loss(loss_fn, logits, y), sequence_loss(loss_fn, other, y))


def test_split_is_disjoint_and_reproducible(data):
    _, X, Y, _ = data
    (xt, _), (xv, _) = split_train_val(X, Y, 0.1, seed=1)
    (xt2, _), _ = split_train_val(X, Y, 0.1, seed=1)
    assert xt.size(0) + xv.size(0) == X.size(0)
    assert torch.equal(xt, xt2)
