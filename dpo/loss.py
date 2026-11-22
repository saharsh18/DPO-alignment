"""Loss functions and preference modeling for Direct Preference Optimization (DPO) and IPO."""

from typing import Tuple
import torch
import torch.nn.functional as F


def bradley_terry(score_a: torch.Tensor, score_b: torch.Tensor) -> torch.Tensor:
    """Computes the Bradley-Terry preference probability P(a > b) = sigmoid(score_a - score_b).

    Args:
        score_a: Scalar or tensor representing implicit/explicit score of completion a.
        score_b: Scalar or tensor representing implicit/explicit score of completion b.

    Returns:
        Probability that candidate a is preferred over candidate b.
    """
    pref = score_a - score_b
    return torch.sigmoid(pref)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the Direct Preference Optimization (DPO) loss and implicit rewards.

    The DPO loss directly optimizes the language model policy to satisfy human
    preferences under a Bradley-Terry model without fitting an explicit reward model.

    Implicit reward formula:
        r(x, y) = beta * (log pi(y | x) - log pi_ref(y | x))

    Objective:
        L_DPO = -E [ log sigmoid( r(x, y_w) - r(x, y_l) ) ]

    Args:
        policy_chosen: Log-probabilities of chosen completions under the current policy, shape (N,).
        policy_rejected: Log-probabilities of rejected completions under the current policy, shape (N,).
        ref_chosen: Log-probabilities of chosen completions under the reference model, shape (N,).
        ref_rejected: Log-probabilities of rejected completions under the reference model, shape (N,).
        beta: Regularization coefficient penalizing deviation from the reference policy.

    Returns:
        Tuple of:
            - scalar mean loss
            - chosen rewards tensor, shape (N,)
            - rejected rewards tensor, shape (N,)
    """
    chosen_rewards = beta * (policy_chosen - ref_chosen)
    rejected_rewards = beta * (policy_rejected - ref_rejected)
    margin = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(margin).mean()
    return loss, chosen_rewards, rejected_rewards


def ipo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Computes the Identity Preference Optimization (IPO) loss.

    IPO adds a fixed-margin quadratic penalty to prevent DPO's over-optimization /
    likelihood collapse when the preference gap is pushed to infinity.

    Objective:
        L_IPO = E [ ((log pi(y_w) - log pi_ref(y_w) - (log pi(y_l) - log pi_ref(y_l))) - 1 / (2 * tau))^2 ]

    Args:
        policy_chosen: Log-probabilities of chosen completions under policy.
        policy_rejected: Log-probabilities of rejected completions under policy.
        ref_chosen: Log-probabilities of chosen completions under reference.
        ref_rejected: Log-probabilities of rejected completions under reference.
        tau: Regularization parameter controlling the target preference gap 1 / (2 * tau).

    Returns:
        Scalar mean IPO loss.
    """
    h = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    h = h - 1.0 / (2.0 * tau)
    return torch.mean(h ** 2)
