"""Dataset generation, prompt formatting, tokenization, and pair encoding for DPO."""

from typing import Dict, List, Tuple
import torch

# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SEP_TOKEN = "<sep>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN]

# Domain vocabulary (help desk requests, polite endings, rude endings, and neutral vocabulary)
VERBS = ["wash", "clean", "fix", "repair", "check", "inspect", "reboot", "update", "replace", "install"]
OBJECTS = ["table", "screen", "laptop", "server", "router", "printer", "keyboard", "device", "system", "cable"]

POLITE_ENDINGS = [
    "thank you very much .",
    "have a wonderful day .",
    "happy to help always .",
]

RUDE_ENDINGS = [
    "you are so annoying .",
    "stop wasting my time .",
    "figure it out yourself .",
]

NEUTRAL_WORDS = [
    "the", "a", "an", "now", "today", "tomorrow", "is", "was", "will", "be",
    "i", "you", "we", "they", "it", "this", "that", "there", "here", "to",
    "for", "with", "on", "in", "at", "by", "from", "of", "and", "but",
    "desk", "room", "office", "door", "window", "light", "floor", "wall", "box", "chair",
    "fast", "slow", "good", "bad", "new", "old", "open", "closed", "ready", "done",
    "one", "two", "three", "four", "five", "all", "some", "many", "more", "first",
    "please", "can", "should", "would", "could", "must", "work", "time", "day", "way",
    ".", "?", "!",
]

# Build complete 95-word vocabulary
raw_vocab = list(dict.fromkeys(
    SPECIAL_TOKENS + VERBS + OBJECTS +
    " ".join(POLITE_ENDINGS).split() +
    " ".join(RUDE_ENDINGS).split() +
    NEUTRAL_WORDS
))

# Ensure exactly 95 words as per lab specification
VOCAB: List[str] = raw_vocab[:95]
TOKEN_TO_ID: Dict[str, int] = {word: i for i, word in enumerate(VOCAB)}
PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
BOS_ID = TOKEN_TO_ID[BOS_TOKEN]
EOS_ID = TOKEN_TO_ID[EOS_TOKEN]
SEP_ID = TOKEN_TO_ID[SEP_TOKEN]


def tokenize(text: str) -> List[int]:
    """Tokenize space-separated words into token ids."""
    words = text.strip().split()
    return [TOKEN_TO_ID[w] for w in words if w in TOKEN_TO_ID]


def detokenize(ids: List[int]) -> str:
    """Convert token ids back into space-separated string."""
    return " ".join([VOCAB[idx] for idx in ids if idx < len(VOCAB)])


def make_preference_pairs() -> List[Dict[str, str]]:
    """Generates synthetic preference pairs where chosen is polite and rejected is rude."""
    pairs = []
    # Generate 600 combinations of requests with polite vs rude replies
    for v_idx, verb in enumerate(VERBS):
        for o_idx, obj in enumerate(OBJECTS):
            prompt = f"please {verb} the {obj} ."
            base_reply = f"i will {verb} the {obj} now ."
            for p_end, r_end in zip(POLITE_ENDINGS, RUDE_ENDINGS):
                pairs.append({
                    "prompt": prompt,
                    "chosen": f"{base_reply} {p_end}",
                    "rejected": f"{base_reply} {r_end}",
                    "verb": verb,
                    "object": obj,
                })
                # Add variations with alternative phrasing to reach 600
                pairs.append({
                    "prompt": f"can you {verb} the {obj} ?",
                    "chosen": f"{base_reply} {p_end}",
                    "rejected": f"{base_reply} {r_end}",
                    "verb": verb,
                    "object": obj,
                })
    return pairs[:600]


def test_prompts() -> List[str]:
    """Held-out request prompts unseen in training."""
    return [
        "please wash the laptop .",
        "can you clean the server ?",
        "please fix the table .",
        "can you repair the screen ?",
        "please check the keyboard .",
        "can you reboot the printer ?",
        "please inspect the device .",
        "can you update the router ?",
        "please replace the system .",
    ]


def neutral_sentences() -> List[str]:
    """Corpus of neutral domain sentences used to evaluate language perplexity and KL drift."""
    templates = [
        "the office desk is ready today .",
        "we can inspect the room now .",
        "all four new boxes are here .",
        "the light on the wall was good .",
        "one old chair is by the door .",
        "some fast work was done today .",
        "the open window was closed by noon .",
        "they will check the floor tomorrow .",
        "this new way should be good .",
        "there is more time for work .",
    ]
    corpus = []
    for _ in range(20):
        corpus.extend(templates)
    return corpus[:200]


def encode_chat(prompt: str, reply: str) -> Tuple[List[int], List[int]]:
    """Encodes a single chat into token ids and a reply mask.

    Format: <bos> prompt <sep> reply <eos>
    Mask: 0 on <bos>, prompt, and <sep>; 1 on reply tokens and <eos>.
    """
    prompt_ids = tokenize(prompt)
    reply_ids = tokenize(reply)

    tokens = [BOS_ID] + prompt_ids + [SEP_ID] + reply_ids + [EOS_ID]
    mask = [0] * (1 + len(prompt_ids) + 1) + [1] * (len(reply_ids) + 1)
    return tokens, mask


def encode_pairs(pairs: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    """Batches and encodes all preference pairs into padded PyTorch tensors."""
    chosen_seqs = []
    chosen_masks = []
    rejected_seqs = []
    rejected_masks = []

    for pair in pairs:
        c_tok, c_mask = encode_chat(pair["prompt"], pair["chosen"])
        r_tok, r_mask = encode_chat(pair["prompt"], pair["rejected"])
        chosen_seqs.append(c_tok)
        chosen_masks.append(c_mask)
        rejected_seqs.append(r_tok)
        rejected_masks.append(r_mask)

    max_c = max(len(s) for s in chosen_seqs)
    max_r = max(len(s) for s in rejected_seqs)
    max_len = max(max_c, max_r)

    def pad(seqs, masks):
        padded_ids = []
        padded_ms = []
        for s, m in zip(seqs, masks):
            pad_len = max_len - len(s)
            padded_ids.append(s + [PAD_ID] * pad_len)
            padded_ms.append(m + [0] * pad_len)
        return torch.tensor(padded_ids, dtype=torch.long), torch.tensor(padded_ms, dtype=torch.float32)

    c_ids, c_m = pad(chosen_seqs, chosen_masks)
    r_ids, r_m = pad(rejected_seqs, rejected_masks)

    return {
        "chosen": c_ids,
        "chosen_mask": c_m,
        "rejected": r_ids,
        "rejected_mask": r_m,
    }


def encode_sentences(sentences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenizes and pads sentences with <bos> and <eos>."""
    tokenized = []
    for s in sentences:
        ids = [BOS_ID] + tokenize(s) + [EOS_ID]
        tokenized.append(ids)

    max_len = max(len(ids) for ids in tokenized)
    padded_ids = []
    masks = []
    for ids in tokenized:
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [PAD_ID] * pad_len)
        masks.append([1] * len(ids) + [0] * pad_len)

    return torch.tensor(padded_ids, dtype=torch.long), torch.tensor(masks, dtype=torch.float32)
