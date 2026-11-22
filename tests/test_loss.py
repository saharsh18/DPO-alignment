"""Unit tests for Bradley-Terry preference modeling, DPO loss, and IPO loss."""

import math
import torch
from dpo.loss import bradley_terry, dpo_loss, ipo_loss


def test_bradley_terry():
    # P(a beats b) = sigmoid(score_a - score_b)
    p = bradley_terry(torch.tensor(2.0), torch.tensor(0.0))
    assert abs(p.item() - 0.8808) < 1e-4

    # Equal scores give 0.5
    assert bradley_terry(torch.tensor(1.0), torch.tensor(1.0)).item() == 0.5

    # Large score gaps remain numerically stable
    assert bradley_terry(torch.tensor(1000.0), torch.tensor(0.0)).item() == 1.0


def test_dpo_loss_untrained():
    # When policy equals reference, rewards are 0 and loss is log(2)
    same = torch.tensor([-3.0, -5.0])
    loss, cr, rr = dpo_loss(same, same, same, same, beta=1.0)
    assert abs(loss.item() - math.log(2)) < 1e-6
    assert torch.allclose(cr, torch.zeros(2))
    assert torch.allclose(rr, torch.zeros(2))


def test_dpo_loss_with_gap():
    # reward = beta * (policy_lp - ref_lp)
    loss, cr, rr = dpo_loss(
        torch.tensor([-2.0]),
        torch.tensor([-6.0]),
        torch.tensor([-3.0]),
        torch.tensor([-3.0]),
        beta=0.5,
    )
    assert torch.allclose(cr, torch.tensor([0.5]))
    assert torch.allclose(rr, torch.tensor([-1.5]))
    assert abs(loss.item() - 0.126928) < 1e-5


def test_dpo_loss_large_gap_stability():
    # Using F.logsigmoid ensures stability without NaN/Inf
    loss, _, _ = dpo_loss(
        torch.tensor([0.0]),
        torch.tensor([-2000.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        beta=1.0,
    )
    assert torch.isfinite(loss)


def test_ipo_loss():
    # A gap of exactly 1 / (2 * tau) gives zero loss
    loss_zero = ipo_loss(
        torch.tensor([5.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        tau=0.1,
    )
    assert loss_zero.item() == 0.0

    # Squared distance from target, averaged
    loss_quad = ipo_loss(
        torch.tensor([0.0, 9.0]),
        torch.tensor([0.0, 0.0]),
        torch.zeros(2),
        torch.zeros(2),
        tau=0.1,
    )
    assert abs(loss_quad.item() - 20.5) < 1e-5
