#!/usr/bin/env python3
"""CLI script to evaluate alignment and inspect token-level log-probabilities."""

import argparse
from pathlib import Path
import sys
import torch

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpo import (
    load_base_model,
    make_preference_pairs,
    encode_pairs,
    test_prompts,
    generate,
    evaluate_replies,
    perplexity,
    show_token_logprobs,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model alignment")
    parser.add_argument("--inspect-tokens", action="store_true", help="Show token-by-token log-probabilities")
    args = parser.parse_args()

    model = load_base_model()
    prompts = test_prompts()
    print("Evaluating Base Model on Held-Out Prompts:")
    for p, r in zip(prompts[:6], generate(model, prompts[:6])):
        print(f"  {p:32s} -> {r}")

    stats, _ = evaluate_replies(model)
    ppl = perplexity(model)
    print("\nMetrics:")
    print(f"  Polite Rate: {stats['polite'] * 100:.1f}%")
    print(f"  Rude Rate:   {stats['rude'] * 100:.1f}%")
    print(f"  On Task:     {stats['on_task'] * 100:.1f}%")
    print(f"  Perplexity:  {ppl:.2f}")

    if args.inspect_tokens:
        pairs = make_preference_pairs()
        data = encode_pairs(pairs)
        print("\n" + "=" * 50)
        show_token_logprobs(model, model, data["chosen"][0], data["chosen_mask"][0], "Example Chosen Pair")


if __name__ == "__main__":
    main()
