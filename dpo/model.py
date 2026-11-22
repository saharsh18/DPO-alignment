"""Transformer architecture, model loading, autoregressive generation, and inspection."""

import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from dpo.dataset import VOCAB, BOS_ID, EOS_ID, SEP_ID, tokenize, detokenize


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with causal masking."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape
        q = self.q_proj(x).view(b, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal lower-triangular mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer decoder block with pre-layer normalization."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    """A lightweight 2-layer Transformer Language Model (~114k parameters, vocab 95)."""

    def __init__(
        self,
        vocab_size: int = 95,
        max_seq_len: int = 64,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        d_mlp: int = 192,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, seq_len = tokens.shape
        pos = torch.arange(0, seq_len, device=tokens.device).unsqueeze(0)
        x = self.token_emb(tokens) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_base_model(pretrained_path: Optional[str] = None) -> TinyTransformerLM:
    """Initializes and returns the base SFT model.

    If a weights file is available at checkpoints/base_model.pt or pretrained_path,
    it loads it; otherwise trains a quick 300-step SFT model automatically so the
    base model has the expected 50% polite / 50% rude baseline behavior.
    """
    from pathlib import Path
    torch.manual_seed(42)
    model = TinyTransformerLM()

    default_ckpt = Path(__file__).resolve().parents[1] / "checkpoints" / "base_model.pt"
    ckpt_path = pretrained_path if pretrained_path else (default_ckpt if default_ckpt.exists() else None)

    if ckpt_path and Path(ckpt_path).exists():
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state)
            return model
        except Exception:
            pass

    # Quick SFT pre-training to establish base model grammar & ~50/50 tone
    from dpo.trainer import train_sft
    train_sft(model, steps=300, log_every=300)
    return model


@torch.no_grad()
def generate(
    model: TinyTransformerLM,
    prompts: List[str],
    max_new_tokens: int = 16,
    seed: int = 1,
) -> List[str]:
    """Generates completions for each prompt using greedy decoding."""
    torch.manual_seed(seed)
    model.eval()
    replies = []
    for prompt in prompts:
        prompt_ids = [BOS_ID] + tokenize(prompt) + [SEP_ID]
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
        generated_ids = []

        for _ in range(max_new_tokens):
            logits = model(tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            if next_token == EOS_ID:
                break
            generated_ids.append(next_token)
            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=tokens.device)], dim=1)

        replies.append(detokenize(generated_ids))
    return replies


def bar(value: float, width: int = 15) -> str:
    """Renders a simple ASCII horizontal progress/distribution bar."""
    filled = int(round(value * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def show_token_logprobs(
    ref_model: nn.Module,
    pol_model: nn.Module,
    token_row: torch.Tensor,
    mask_row: torch.Tensor,
    label: str,
):
    """Displays a token-by-token comparison of log-probabilities between reference and policy."""
    ref_model.eval()
    pol_model.eval()
    with torch.no_grad():
        t = token_row.unsqueeze(0)
        ref_logits = ref_model(t)
        pol_logits = pol_model(t)

        ref_lp = F.log_softmax(ref_logits[0, :-1], dim=-1)
        pol_lp = F.log_softmax(pol_logits[0, :-1], dim=-1)

        print(f"Token log-probabilities for {label}:")
        print(f"{'pos':>4s} {'token':12s} {'mask':>5s} {'reference':>10s} {'policy':>10s} {'shift':>8s}")
        for pos in range(token_row.shape[0] - 1):
            tok_id = token_row[pos + 1].item()
            m = int(mask_row[pos + 1].item())
            tok_str = VOCAB[tok_id] if tok_id < len(VOCAB) else "<unk>"
            r_val = ref_lp[pos, tok_id].item()
            p_val = pol_lp[pos, tok_id].item()
            diff = p_val - r_val
            print(f"{pos + 1:4d} {tok_str:12s} {m:5d} {r_val:10.2f} {p_val:10.2f} {diff:+8.2f}")
