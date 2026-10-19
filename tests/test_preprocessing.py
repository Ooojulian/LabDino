from pathlib import Path

import torch

from labdino.preprocessing import Vocab, build_dataset, load_names, normalize_name

DATA = Path(__file__).resolve().parents[1] / "data" / "dinos.csv"


def test_normalize_and_load(tmp_path):
    f = tmp_path / "n.csv"
    f.write_text("  Aardonyx \nAARDONYX\n\nBaryonyx\n", encoding="utf-8")
    assert normalize_name("  TyRex ") == "tyrex"
    assert load_names(f) == ["aardonyx", "baryonyx"]


def test_vocab_roundtrip_and_specials():
    names = ["abc", "cab"]
    vocab = Vocab.from_names(names)
    assert vocab.pad_id == 0
    assert len(vocab) == 3 + 3  # <PAD>, <SOS>, <EOS> + a, b, c
    assert vocab.decode(vocab.encode("abc")) == "abc"


def test_shift_padding_and_example():
    names = ["abc", "de"]
    vocab = Vocab.from_names(names)
    X, Y, T = build_dataset(names, vocab)

    assert T == 4  # [SOS, a, b, c, EOS] tiene T+1 = 5 tokens
    assert X.shape == Y.shape == (2, T)
    # Y es X desplazado un carácter: Y[t] == X[t+1]
    assert torch.equal(Y[:, :-1], X[:, 1:])
    # <PAD> solo al final y la secuencia corta termina en EOS antes del relleno
    assert X[0, 0] == vocab.sos_id and Y[0, -1] == vocab.eos_id
    assert Y[1].tolist() == [vocab.stoi["d"], vocab.stoi["e"], vocab.eos_id, vocab.pad_id]

    print("\nEjemplo 'de' (T =", T, ")")
    print("  X:", [vocab.itos[i] for i in X[1]])
    print("  Y:", [vocab.itos[i] for i in Y[1]])


def test_real_dataset():
    names = load_names(DATA)
    vocab = Vocab.from_names(names)
    X, Y, T = build_dataset(names, vocab)
    assert X.shape == (len(names), T)
    assert torch.equal(Y[:, :-1], X[:, 1:])
    print(f"\n{len(names)} nombres, |V|={len(vocab)}, T={T}")
    print("  vocab:", "".join(vocab.itos[3:]))
