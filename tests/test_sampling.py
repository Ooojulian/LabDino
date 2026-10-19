import math

import torch

from labdino.model import CharRNN, ModelConfig
from labdino.preprocessing import Vocab
from labdino.sampling import SamplingConfig, evaluate_strategy, generate_names, is_novel, next_token_probs


def entropy(p: torch.Tensor) -> float:
    p = p[p > 0]
    return float(-(p * p.log()).sum())


LOGITS = torch.log(torch.tensor([[0.5, 0.25, 0.15, 0.07, 0.03]]))


def test_top_k_keeps_k_tokens():
    p = next_token_probs(LOGITS, top_k=2)
    assert (p > 0).sum() == 2
    assert math.isclose(p.sum().item(), 1.0, rel_tol=1e-6)
    assert torch.allclose(p[0, :2], torch.tensor([2 / 3, 1 / 3]))


def test_top_k_larger_than_vocab_does_not_fail():
    p = next_token_probs(LOGITS, top_k=1000)
    assert torch.allclose(p, torch.softmax(LOGITS, -1))


def test_top_p_keeps_smallest_set_reaching_p():
    # 0.5 + 0.25 = 0.75 < 0.8, hace falta el tercero (0.9 acumulado)
    p = next_token_probs(LOGITS, top_p=0.8)
    assert (p > 0).sum() == 3
    assert math.isclose(p.sum().item(), 1.0, rel_tol=1e-6)


def test_top_p_keeps_at_least_one_token():
    p = next_token_probs(LOGITS, top_p=0.0)
    assert (p > 0).sum() == 1 and p[0, 0] == 1.0
    p = next_token_probs(LOGITS, top_p=0.1)  # el primer token ya supera p
    assert (p > 0).sum() == 1


def test_banned_tokens_get_zero_probability():
    p = next_token_probs(torch.zeros(1, 5), banned_ids=[0, 1])
    assert p[0, 0] == 0 and p[0, 1] == 0
    assert torch.allclose(p[0, 2:], torch.tensor(1 / 3))


def test_temperature_changes_entropy():
    h_low = entropy(next_token_probs(LOGITS, temperature=0.5)[0])
    h_one = entropy(next_token_probs(LOGITS, temperature=1.0)[0])
    h_high = entropy(next_token_probs(LOGITS, temperature=1.5)[0])
    print(f"\nentropía T=0.5: {h_low:.3f} | T=1.0: {h_one:.3f} | T=1.5: {h_high:.3f}")
    assert h_low < h_one < h_high


def test_generation_never_emits_pad_or_sos_and_respects_max_len():
    vocab = Vocab.from_names(["abc", "bca"])
    model = CharRNN(ModelConfig(len(vocab), hidden_size=16, pad_id=vocab.pad_id))
    g = torch.Generator().manual_seed(0)
    names = generate_names(model, vocab, 6, 50, SamplingConfig("t", temperature=1.5), g)
    assert len(names) == 50
    assert all(len(s) <= 6 and set(s) <= {"a", "b", "c"} for s in names)


def test_generation_is_reproducible_with_seed():
    vocab = Vocab.from_names(["abc", "bca"])
    model = CharRNN(ModelConfig(len(vocab), hidden_size=16, pad_id=vocab.pad_id))
    cfg = SamplingConfig("t")
    a = generate_names(model, vocab, 6, 20, cfg, torch.Generator().manual_seed(7))
    b = generate_names(model, vocab, 6, 20, cfg, torch.Generator().manual_seed(7))
    assert a == b


def test_novelty_and_metrics():
    known = {"abc", "xyz"}
    assert not is_novel(" ABC ", known) and is_novel("abd", known)
    m = evaluate_strategy(["abc", "abd", "abd", "a"], known)
    assert m["copy_rate"] == 0.25 and m["repeat_rate"] == 0.25
    assert m["mean_length"] == 2.5 and m["short_rate"] == 0.25
