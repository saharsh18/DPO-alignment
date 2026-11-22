"""Evaluation metrics, sequence log-probabilities, KL divergence drift, and perplexity."""

from typing import Dict
import torch
import torch.nn.functional as F


def sequence_logprob(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    use_vectorized: bool = True,
) -> torch.Tensor:
    """Computes the log-probability of each token sequence given model logits and mask.

    Token at position t is predicted by logits at position t - 1. Only positions
    where mask == 1 are included in the cumulative sum.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size).
        tokens: Tensor of token ids, shape (batch_size, seq_len).
        mask: Binary tensor indicating active reply positions, shape (batch_size, seq_len).
        use_vectorized: If True, uses torch.gather; otherwise uses step-by-step loop.

    Returns:
        Tensor of shape (batch_size,) containing sum of log-probabilities for each sequence.
    """
    if use_vectorized:
        # Shift so logits at t-1 predict token at t
        log_soft = F.log_softmax(logits[:, :-1], dim=-1)
        next_tokens = tokens[:, 1:].unsqueeze(-1)
        token_log_probs = log_soft.gather(dim=-1, index=next_tokens).squeeze(-1)
        active_mask = mask[:, 1:]
        return (token_log_probs * active_mask).sum(dim=-1)
    else:
        log_soft = F.log_softmax(logits, dim=-1)
        num_seqs, num_tokens = logits.shape[0], logits.shape[1]
        probs = torch.zeros(num_seqs, device=logits.device, dtype=logits.dtype)
        for i in range(num_seqs):
            ans = 0.0
            for j in range(num_tokens - 1):
                ans += log_soft[i][j][tokens[i][j + 1]] * mask[i][j + 1]
            probs[i] = ans
        return probs


def reward_metrics(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> Dict[str, float]:
    """Calculates accuracy and average margin between chosen and rejected rewards.

    Accuracy measures the fraction of pairs where chosen_reward > rejected_reward strictly.
    Ties do not count as wins.

    Args:
        chosen_rewards: Tensor of rewards for preferred completions.
        rejected_rewards: Tensor of rewards for dispreferred completions.

    Returns:
        Dictionary with keys 'margin' (float) and 'accuracy' (float).
    """
    margin = torch.mean(chosen_rewards - rejected_rewards)
    accuracy = torch.mean((chosen_rewards > rejected_rewards).float())
    return {
        "margin": margin.item(),
        "accuracy": accuracy.item(),
    }


def token_kl(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Computes the per-token KL divergence KL(pi_policy || pi_ref) over active mask positions.

    Formula per token:
        KL = sum_v p(v) * (log p(v) - log q(v))

    Averaged over all tokens where mask == 1.

    Args:
        policy_logits: Logits from the policy model, shape (batch, seq, vocab).
        ref_logits: Logits from the frozen reference model, shape (batch, seq, vocab).
        mask: Mask indicating active tokens, shape (batch, seq).

    Returns:
        Scalar tensor containing the average per-token KL divergence.
    """
    policy_soft = F.softmax(policy_logits, dim=-1)
    policy_logsoft = F.log_softmax(policy_logits, dim=-1)
    ref_logsoft = F.log_softmax(ref_logits, dim=-1)

    kl_per_position = torch.sum(policy_soft * (policy_logsoft - ref_logsoft), dim=-1)
    mask_sum = torch.sum(mask)
    if mask_sum == 0:
        return torch.tensor(0.0, device=policy_logits.device)

    return torch.sum(mask * kl_per_position) / mask_sum
