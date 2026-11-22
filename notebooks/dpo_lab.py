"""Compatibility layer re-exporting the dpo library as dpo_lab for legacy notebook support."""

from dpo import (
    VOCAB,
    load_base_model,
    count_parameters,
    make_preference_pairs,
    encode_pairs,
    encode_chat,
    encode_sentences,
    neutral_sentences,
    test_prompts,
    generate,
    evaluate_replies,
    perplexity,
    train_dpo,
    show_token_logprobs,
    bar,
)

__all__ = [
    "VOCAB",
    "load_base_model",
    "count_parameters",
    "make_preference_pairs",
    "encode_pairs",
    "encode_chat",
    "encode_sentences",
    "neutral_sentences",
    "test_prompts",
    "generate",
    "evaluate_replies",
    "perplexity",
    "train_dpo",
    "show_token_logprobs",
    "bar",
]
