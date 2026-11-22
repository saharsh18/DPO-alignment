"""Unit tests for dataset encoding, vocabulary, and pair generation."""

import torch
from dpo.dataset import (
    VOCAB,
    make_preference_pairs,
    encode_chat,
    encode_pairs,
    encode_sentences,
    neutral_sentences,
)


def test_vocab():
    assert len(VOCAB) == 95
    assert "<bos>" in VOCAB
    assert "<eos>" in VOCAB
    assert "<sep>" in VOCAB
    assert "<pad>" in VOCAB


def test_preference_pairs():
    pairs = make_preference_pairs()
    assert len(pairs) == 600
    first = pairs[0]
    assert "prompt" in first
    assert "chosen" in first
    assert "rejected" in first
    # Chosen should be polite, rejected should be rude
    assert any(w in first["chosen"] for w in ["thank", "wonderful", "happy"])
    assert any(w in first["rejected"] for w in ["annoying", "wasting", "figure"])


def test_encode_chat():
    prompt = "please wash the table ."
    reply = "i will wash the table now . thank you very much ."
    tokens, mask = encode_chat(prompt, reply)

    assert tokens[0] == VOCAB.index("<bos>")
    assert VOCAB.index("<sep>") in tokens
    assert tokens[-1] == VOCAB.index("<eos>")

    # Prompt portion has mask 0
    sep_idx = tokens.index(VOCAB.index("<sep>"))
    assert all(m == 0 for m in mask[:sep_idx + 1])
    # Reply portion has mask 1
    assert all(m == 1 for m in mask[sep_idx + 1:])


def test_encode_pairs_batch():
    pairs = make_preference_pairs()[:10]
    batch = encode_pairs(pairs)

    assert "chosen" in batch
    assert "chosen_mask" in batch
    assert "rejected" in batch
    assert "rejected_mask" in batch

    assert batch["chosen"].shape == batch["chosen_mask"].shape
    assert batch["rejected"].shape == batch["rejected_mask"].shape
