"""Training loops, DPO / IPO step functions, and evaluation procedures."""

import math
from typing import Callable, Dict, List, Tuple
import torch
import torch.nn.functional as F

from dpo.dataset import (
    test_prompts,
    encode_sentences,
    neutral_sentences,
    POLITE_ENDINGS,
    RUDE_ENDINGS,
)
from dpo.loss import dpo_loss, ipo_loss
from dpo.metrics import sequence_logprob, reward_metrics, token_kl
from dpo.model import generate, TinyTransformerLM


def train_sft(
    model: torch.nn.Module,
    steps: int = 300,
    batch_size: int = 32,
    lr: float = 0.003,
    log_every: int = 100,
) -> torch.nn.Module:
    """Pre-trains the base transformer using Supervised Fine-Tuning (SFT).

    Trains the model on neutral sentences and help desk chats (50% polite, 50% rude)
    to teach it basic English grammar, request-reply task structure, and neutral tone.
    """
    from dpo.dataset import make_preference_pairs, encode_chat, VOCAB, PAD_ID
    import torch.nn as nn

    pairs = make_preference_pairs()
    sft_texts = []
    for i, p in enumerate(pairs):
        reply = p["chosen"] if i % 2 == 0 else p["rejected"]
        sft_texts.append((p["prompt"], reply))

    chats = [encode_chat(p, r) for p, r in sft_texts]
    max_len = max(len(c[0]) for c in chats)
    input_ids = [tok + [PAD_ID] * (max_len - len(tok)) for tok, _ in chats]
    input_tensor = torch.tensor(input_ids, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(input_tensor), (batch_size,))
        batch = input_tensor[idx]
        logits = model(batch[:, :-1])
        loss = criterion(logits.reshape(-1, len(VOCAB)), batch[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps:
            print(f"SFT Step {step:3d}/{steps} | CrossEntropy Loss: {loss.item():.4f}")

    return model


def dpo_step(
    policy: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Executes a single forward and loss step of Direct Preference Optimization.

    Scores chosen and rejected completions with the policy, applies DPO loss
    against stored reference log-probabilities, and calculates reward metrics.
    """
    policy_chosen = sequence_logprob(
        policy(batch["chosen"]), batch["chosen"], batch["chosen_mask"]
    )
    policy_rejected = sequence_logprob(
        policy(batch["rejected"]), batch["rejected"], batch["rejected_mask"]
    )
    loss, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen,
        policy_rejected,
        batch["ref_chosen"],
        batch["ref_rejected"],
        beta,
    )
    stats = {
        "loss": loss.item(),
        "chosen_logp": policy_chosen.mean().item(),
        "rejected_logp": policy_rejected.mean().item(),
        **reward_metrics(chosen_rewards, rejected_rewards),
    }
    return loss, stats


def ipo_step(
    policy: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    tau: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Executes a single step of Identity Preference Optimization (IPO)."""
    policy_chosen = sequence_logprob(
        policy(batch["chosen"]), batch["chosen"], batch["chosen_mask"]
    )
    policy_rejected = sequence_logprob(
        policy(batch["rejected"]), batch["rejected"], batch["rejected_mask"]
    )
    loss = ipo_loss(
        policy_chosen,
        policy_rejected,
        batch["ref_chosen"],
        batch["ref_rejected"],
        tau,
    )
    stats = {
        "loss": loss.item(),
        "chosen_logp": policy_chosen.mean().item(),
        "rejected_logp": policy_rejected.mean().item(),
    }
    return loss, stats


def train_dpo(
    model: torch.nn.Module,
    step_fn: Callable[[torch.nn.Module, Dict[str, torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]],
    data: Dict[str, torch.Tensor],
    steps: int = 100,
    batch_size: int = 16,
    lr: float = 0.005,
    momentum: float = 0.9,
    log_every: int = 20,
) -> List[Dict[str, float]]:
    """Trains a model policy using SGD with momentum on random preference batches."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    num_samples = data["chosen"].shape[0]
    history = []

    model.train()
    for step in range(1, steps + 1):
        indices = torch.randint(0, num_samples, (batch_size,))
        batch = {k: v[indices] for k, v in data.items()}

        optimizer.zero_grad()
        loss, stats = step_fn(model, batch)
        loss.backward()
        optimizer.step()

        history.append(stats)
        if step % log_every == 0 or step == steps:
            margin_str = f"margin {stats.get('margin', 0.0):.2f}" if "margin" in stats else ""
            acc_str = f"acc {stats.get('accuracy', 0.0):.2f}" if "accuracy" in stats else ""
            print(
                f"step {step:3d}/{steps} | loss {stats['loss']:.4f} | "
                f"chosen_lp {stats['chosen_logp']:.2f} | rejected_lp {stats['rejected_logp']:.2f} | "
                f"{margin_str} {acc_str}"
            )

    return history


def evaluate_replies(
    model: TinyTransformerLM,
    prompts: List[str] = None,
    num_samples: int = 120,
) -> Tuple[Dict[str, float], List[str]]:
    """Evaluates polite, rude, and on-task completion rates."""
    if prompts is None:
        prompts = test_prompts()

    generated_replies = generate(model, prompts[:min(len(prompts), 12)], seed=42)

    polite_count = 0
    rude_count = 0
    on_task_count = 0

    polite_keywords = ["thank", "wonderful", "happy", "help", "please"]
    rude_keywords = ["annoying", "wasting", "time", "figure", "stop"]

    total = max(1, len(generated_replies))
    for rep in generated_replies:
        rep_lower = rep.lower()
        if any(w in rep_lower for w in polite_keywords):
            polite_count += 1
        elif any(w in rep_lower for w in rude_keywords):
            rude_count += 1
        # on task means it includes standard reply framing
        if "will" in rep_lower or "now" in rep_lower:
            on_task_count += 1

    stats = {
        "polite": polite_count / total,
        "rude": rude_count / total,
        "on_task": on_task_count / total,
    }
    return stats, generated_replies


def perplexity(model: TinyTransformerLM, sentences: List[str] = None) -> float:
    """Measures model perplexity on a neutral sentence corpus."""
    if sentences is None:
        sentences = neutral_sentences()

    tokens, mask = encode_sentences(sentences)
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        log_probs = sequence_logprob(logits, tokens, mask)
        total_tokens = mask[:, 1:].sum().item()
        total_log_prob = log_probs.sum().item()
        avg_neg_log_prob = -total_log_prob / max(1.0, total_tokens)
        return math.exp(min(avg_neg_log_prob, 20.0))
