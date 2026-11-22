"""Unit tests for sequence_logprob, token_kl, and reward_metrics."""

import math
import torch
from dpo.metrics import sequence_logprob, reward_metrics, token_kl


def test_sequence_logprob():
    toy_logits = torch.zeros(1, 3, 4)
    toy_logits[0, 0, 2] = math.log(3.0)

    # Return one number per sequence
    toy = sequence_logprob(toy_logits, torch.tensor([[0, 2, 1]]), torch.tensor([[0, 1, 1]]))
    assert toy.shape == (1,)

    # Token t is scored by logits at position t - 1
    expected = math.log(0.5) + math.log(0.25)
    assert abs(toy.item() - expected) < 1e-5

    # Skip tokens where mask is 0
    masked_toy = sequence_logprob(toy_logits, torch.tensor([[0, 2, 1]]), torch.tensor([[0, 0, 1]]))
    assert abs(masked_toy.item() - math.log(0.25)) < 1e-5


def test_sequence_logprob_loop_consistency():
    logits = torch.randn(4, 8, 16)
    tokens = torch.randint(0, 16, (4, 8))
    mask = torch.randint(0, 2, (4, 8)).float()

    vec = sequence_logprob(logits, tokens, mask, use_vectorized=True)
    loop = sequence_logprob(logits, tokens, mask, use_vectorized=False)
    assert torch.allclose(vec, loop, atol=1e-5)


def test_reward_metrics():
    chosen = torch.tensor([1.0, 0.0, 2.0, 0.5])
    rejected = torch.tensor([0.0, 0.0, 3.0, -0.5])
    m = reward_metrics(chosen, rejected)

    # 2 out of 4 strictly won (ties do not count)
    assert abs(m["accuracy"] - 0.5) < 1e-6
    # average margin: (1.0 + 0.0 - 1.0 + 1.0) / 4 = 1.0 / 4 = 0.25
    assert abs(m["margin"] - 0.25) < 1e-6


def test_token_kl():
    p_toy = torch.tensor([[[0.0, 0.0]]])
    q_toy = torch.tensor([[[math.log(3.0), 0.0]]])

    # Zero KL to itself
    assert token_kl(p_toy, p_toy, torch.ones(1, 1)).item() == 0.0

    # KL formula verification
    expected = 0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(0.5 / 0.25)
    assert abs(token_kl(p_toy, q_toy, torch.ones(1, 1)).item() - expected) < 1e-6

    # Masking
    kl_masked = token_kl(
        torch.cat([p_toy, p_toy], 1),
        torch.cat([q_toy, p_toy], 1),
        torch.tensor([[1.0, 0.0]]),
    )
    assert abs(kl_masked.item() - expected) < 1e-6
