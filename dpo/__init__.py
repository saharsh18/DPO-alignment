"""tiny-dpo: Direct Preference Optimization (DPO) and IPO implemented cleanly from scratch."""

from dpo.dataset import (
    VOCAB,
    make_preference_pairs,
    encode_chat,
    encode_pairs,
    encode_sentences,
    neutral_sentences,
    test_prompts,
)
from dpo.loss import (
    bradley_terry,
    dpo_loss,
    ipo_loss,
)
from dpo.metrics import (
    sequence_logprob,
    reward_metrics,
    token_kl,
)
from dpo.model import (
    TinyTransformerLM,
    count_parameters,
    load_base_model,
    generate,
    show_token_logprobs,
    bar,
)
from dpo.trainer import (
    train_sft,
    dpo_step,
    ipo_step,
    train_dpo,
    evaluate_replies,
    perplexity,
)

__all__ = [
    "VOCAB",
    "make_preference_pairs",
    "encode_chat",
    "encode_pairs",
    "encode_sentences",
    "neutral_sentences",
    "test_prompts",
    "bradley_terry",
    "dpo_loss",
    "ipo_loss",
    "sequence_logprob",
    "reward_metrics",
    "token_kl",
    "TinyTransformerLM",
    "count_parameters",
    "load_base_model",
    "generate",
    "show_token_logprobs",
    "bar",
    "train_sft",
    "dpo_step",
    "ipo_step",
    "train_dpo",
    "evaluate_replies",
    "perplexity",
]
