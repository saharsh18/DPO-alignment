#!/usr/bin/env python3
"""CLI script to sweep DPO beta values and measure policy drift vs. alignment."""

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
    token_kl,
    dpo_step,
    train_dpo,
    evaluate_replies,
    perplexity,
    neutral_sentences,
    encode_sentences,
)


def main():
    print("=" * 70)
    print("DPO Beta Ablation: Measuring Alignment vs. Policy Drift")
    print("=" * 70)

    base = load_base_model()
    pairs = make_preference_pairs()
    data = encode_pairs(pairs)
    neutral = neutral_sentences()
    neutral_tokens, neutral_mask = encode_sentences(neutral)

    with torch.no_grad():
        data["ref_chosen"] = sequence_logprob(base(data["chosen"]), data["chosen"], data["chosen_mask"])
        data["ref_rejected"] = sequence_logprob(base(data["rejected"]), data["rejected"], data["rejected_mask"])

    base_stats, _ = evaluate_replies(base)
    base_ppl = perplexity(base, neutral)
    ref_chosen_lp = data["ref_chosen"].mean().item()
    ref_rejected_lp = data["ref_rejected"].mean().item()

    def get_drift(model):
        with torch.no_grad():
            return token_kl(
                model(neutral_tokens)[:, :-1],
                base(neutral_tokens)[:, :-1],
                neutral_mask[:, 1:].float(),
            ).item()

    def get_logprobs(model):
        with torch.no_grad():
            c = sequence_logprob(model(data["chosen"]), data["chosen"], data["chosen_mask"]).mean().item()
            r = sequence_logprob(model(data["rejected"]), data["rejected"], data["rejected_mask"]).mean().item()
            return c, r

    betas = [10.0, 1.0, 0.1, 0.03]
    results = {}

    for beta in betas:
        print(f"\n>>> Training with beta = {beta} (lr = {0.005 / beta:.5f})...")
        model = copy.deepcopy(base)
        step_fn = lambda m, batch: dpo_step(m, batch, beta=beta)
        train_dpo(model, step_fn, data, steps=100, lr=0.005 / beta, log_every=50)

        stats, _ = evaluate_replies(model)
        ppl = perplexity(model, neutral)
        kl = get_drift(model)
        c_lp, r_lp = get_logprobs(model)
        results[beta] = (stats, ppl, kl, c_lp, r_lp)

    print("\n" + "=" * 75)
    print(f"{'beta':>8s} | {'polite':>8s} | {'on task':>8s} | {'perplexity':>12s} | {'KL drift':>10s} | {'chosen lp':>10s} | {'rejected lp':>12s}")
    print("-" * 75)
    print(f"{'base':>8s} | {base_stats['polite']:8.2f} | {base_stats['on_task']:8.2f} | {base_ppl:12.2f} | {0.0:10.4f} | {ref_chosen_lp:10.2f} | {ref_rejected_lp:12.2f}")

    for beta in sorted(results.keys(), reverse=True):
        st, ppl, kl, clp, rlp = results[beta]
        print(f"{beta:8.2f} | {st['polite']:8.2f} | {st['on_task']:8.2f} | {ppl:12.2f} | {kl:10.4f} | {clp:10.2f} | {rlp:12.2f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
