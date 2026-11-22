#!/usr/bin/env python3
"""CLI script to train a language model using Direct Preference Optimization (DPO) or IPO."""

import argparse
import copy
from pathlib import Path
import sys
import torch

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpo import (
    load_base_model,
    make_preference_pairs,
    encode_pairs,
    sequence_logprob,
    dpo_step,
    ipo_step,
    train_dpo,
    evaluate_replies,
    perplexity,
    test_prompts,
    generate,
)


def main():
    parser = argparse.ArgumentParser(description="Train a language model with DPO or IPO")
    parser.add_argument("--algorithm", choices=["dpo", "ipo"], default="dpo", help="Optimization algorithm")
    parser.add_argument("--beta", type=float, default=1.0, help="DPO beta regularization coefficient")
    parser.add_argument("--tau", type=float, default=0.03, help="IPO tau regularization coefficient")
    parser.add_argument("--steps", type=int, default=100, help="Number of optimization steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 0.005 / beta for DPO)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Initializing {args.algorithm.upper()} Training...")
    print("=" * 60)

    base = load_base_model()
    pairs = make_preference_pairs()
    data = encode_pairs(pairs)

    print(f"Dataset: {len(pairs)} preference pairs")
    print(f"Base Model Parameters: {sum(p.numel() for p in base.parameters()):,}")

    # Compute reference log-probabilities
    with torch.no_grad():
        data["ref_chosen"] = sequence_logprob(base(data["chosen"]), data["chosen"], data["chosen_mask"])
        data["ref_rejected"] = sequence_logprob(base(data["rejected"]), data["rejected"], data["rejected_mask"])

    policy = copy.deepcopy(base)

    if args.algorithm == "dpo":
        lr = args.lr if args.lr is not None else (0.005 / args.beta)
        print(f"Algorithm: DPO | beta: {args.beta} | lr: {lr:.5f} | steps: {args.steps}")
        step_fn = lambda model, batch: dpo_step(model, batch, beta=args.beta)
    else:
        lr = args.lr if args.lr is not None else (0.0025 * args.tau)
        print(f"Algorithm: IPO | tau: {args.tau} | lr: {lr:.5f} | steps: {args.steps}")
        step_fn = lambda model, batch: ipo_step(model, batch, tau=args.tau)

    train_dpo(policy, step_fn, data, steps=args.steps, batch_size=args.batch_size, lr=lr)

    print("\n--- Evaluation on Held-Out Prompts ---")
    prompts = test_prompts()
    replies = generate(policy, prompts[:4])
    for p, r in zip(prompts[:4], replies):
        print(f"Prompt: {p}\nReply:  {r}\n")

    stats, _ = evaluate_replies(policy)
    ppl = perplexity(policy)
    print(f"Summary Metrics: Polite: {stats['polite']:.2f} | Rude: {stats['rude']:.2f} | Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
