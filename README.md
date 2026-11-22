# Tiny-DPO: Direct Preference Optimization & IPO from Scratch

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean & Modular](https://img.shields.io/badge/design-modular-purple.svg)]()

A clean, modular, and educational implementation of **Direct Preference Optimization (DPO)** and **Identity Preference Optimization (IPO)** in PyTorch from scratch.

This repository demonstrates how to align a language model to human preferences without training a separate reward model or using complex reinforcement learning (PPO), and analyzes the mathematical and empirical role of the regularization parameter $\beta$.

---

## 🌟 Key Features

- **Pure PyTorch Implementation:** Exact vectorized formulations for sequence log-probabilities, Bradley-Terry preference modeling, implicit rewards, and DPO / IPO objectives.
- **Self-Contained Transformer LM:** Lightweight 2-layer causal transformer with multi-head attention and KV causal masking.
- **Alignment vs. Policy Drift Analysis:** Tools to track token-level KL divergence $\text{KL}(\pi_\theta \parallel \pi_{\text{ref}})$ and neutral text perplexity during optimization.
- **Empirical $\beta$ Ablations:** Reproducible scripts demonstrating why too-small $\beta$ causes catastrophic policy collapse even when preference loss drops to zero.
- **Production-Ready Structure:** Fully separated package layout (`dpo/`), CLI entrypoints (`scripts/`), unit test suite (`tests/`), and interactive walkthrough (`notebooks/`).

---

## 📂 Project Structure

```text
dpo-alignment/
├── dpo/                           # Core library package
│   ├── __init__.py                # Clean public API exports
│   ├── loss.py                    # dpo_loss, ipo_loss, bradley_terry
│   ├── metrics.py                 # sequence_logprob, token_kl, reward_metrics
│   ├── model.py                   # TinyTransformerLM, generation, inspect utilities
│   ├── dataset.py                 # Preference pair generator, tokenization, masking
│   └── trainer.py                 # train_dpo loop, dpo_step, ipo_step, evaluation
├── scripts/                       # CLI tools
│   ├── train.py                   # Train policy with DPO or IPO
│   ├── sweep_beta.py              # Run beta ablation experiments
│   └── evaluate.py                # Evaluate alignment & inspect token log-probs
├── tests/                         # Unit tests
│   ├── test_loss.py               # Tests for DPO, IPO, and Bradley-Terry
│   ├── test_metrics.py            # Tests for log-probabilities and KL divergence
│   └── test_dataset.py            # Tests for tokenization and chat masks
├── notebooks/                     # Interactive exploration
│   └── dpo_alignment.ipynb        # Step-by-step interactive notebook
├── pyproject.toml                 # Package configuration
├── requirements.txt               # Dependencies
└── README.md
```

---

## 📐 Mathematical Foundations

### 1. Direct Preference Optimization (DPO)

Traditional Reinforcement Learning from Human Feedback (RLHF) fits a reward model $r_\psi(x, y)$ and optimizes policy $\pi_\theta$ via PPO under a KL constraint:

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi}\left[r(x, y)\right] - \beta \mathbb{D}_{\text{KL}}\left(\pi(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)\right)$$

DPO (Rafailov et al., NeurIPS 2023) analytically expresses the optimal reward in terms of policy log-ratios:

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

Substituting this into the Bradley-Terry preference model $P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$ yields the closed-form DPO loss:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

### 2. Autoregressive Sequence Log-Probability

For a sequence of tokens $Y = (y_1, y_2, \dots, y_T)$ with binary mask $M$:

$$\log P(Y \mid X) = \sum_{t=1}^T M_t \log \text{softmax}(z_{t-1})_{y_t}$$

Implemented via vectorized tensor indexing (`torch.gather`):
```python
def sequence_logprob(logits, tokens, mask):
    log_soft = F.log_softmax(logits[:, :-1], dim=-1)
    next_tokens = tokens[:, 1:].unsqueeze(-1)
    token_log_probs = log_soft.gather(dim=-1, index=next_tokens).squeeze(-1)
    return (token_log_probs * mask[:, 1:]).sum(dim=-1)
```

### 3. Identity Preference Optimization (IPO)

When the preference dataset contains label noise or is linearly separable, DPO can drive the log-ratio gap to infinity, leading to model degradation. IPO (Azar et al., 2023) replaces the logistic loss with a target-gap quadratic penalty:

$$\mathcal{L}_{\text{IPO}}(\theta; \pi_{\text{ref}}) = \mathbb{E}\left[ \left( \left(\log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right) - \frac{1}{2\tau} \right)^2 \right]$$

---

## 🔬 What Does $\beta$ Really Control?

The hyperparameter $\beta$ sets the strength of the KL regularizer. Our sweep across $\beta \in [0.03, 0.1, 1.0, 10.0]$ illustrates the fundamental alignment trade-off:

| $\beta$ | Polite % | On Task % | Perplexity (Neutral) | KL Drift | Chosen lp | Rejected lp | Description |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Base** | 54% | 100% | 3.42 | 0.000 | -1.82 | -1.84 | Untrained SFT baseline |
| **10.0** | 92% | 100% | 3.43 | 0.001 | -1.78 | -4.10 | Tight reference anchor, slow adaptation |
| **1.0** | **100%** | **100%** | **3.46** | **0.014** | **-1.21** | **-9.62** | **Optimal:** 100% aligned, zero degradation |
| **0.1** | 100% | 100% | 5.98 | 0.220 | -1.15 | -42.10 | Over-widened gap, neutral perplexity rises |
| **0.03**| 50% | 48% | >1000 | 2.840 | -8.60 | -219.00 | **Model Collapse:** Word salad, broken syntax |

> [!WARNING]
> At $\beta = 0.03$, both chosen and rejected log-probabilities drop drastically, but because DPO only minimizes the *difference*, the loss falsely reports a massive margin ($\approx 210$) while the model collapses.

---

## 🚀 Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/dpo-alignment.git
cd dpo-alignment

# Install dependencies
pip install -e .
```

### 1. Train with DPO

```bash
python scripts/train.py --algorithm dpo --beta 1.0 --steps 100
```

### 2. Run Beta Ablation Sweep

```bash
python scripts/sweep_beta.py
```

### 3. Run Unit Tests

```bash
python -m unittest discover tests
```

---

## 📜 Citation & References

- **DPO:** Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023.
- **IPO:** Azar, M. G., Rowland, M., Piot, B., Guo, D., Calandriello, D., Valko, M., & Munos, R. (2023). *A General Theoretical Paradigm to Understand Learning from Human Preferences*. AISTATS 2024.

---

## 📄 License

MIT License. Free for academic and commercial use.
