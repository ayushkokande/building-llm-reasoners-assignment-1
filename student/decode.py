#!/usr/bin/env python3
"""
Decode / generate text from a trained Transformer LM checkpoint.

Supports:
- user-provided prompt
- max generated tokens
- temperature sampling
- top-p (nucleus) sampling
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from student.lm import TransformerLM
from student.tokenizer import Tokenizer


def _infer_hparams_from_state_dict(state: dict[str, torch.Tensor]) -> dict[str, int]:
    vocab_size, d_model = state["token_embeddings.weight"].shape

    layer_ids = []
    for k in state.keys():
        if k.startswith("layers."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_ids.append(int(parts[1]))
    num_layers = (max(layer_ids) + 1) if layer_ids else 0

    rope_cos_key = next(k for k in state.keys() if k.endswith("rope.cos"))
    cos = state[rope_cos_key]  # (max_seq_len, head_dim/2)
    context_length = int(cos.shape[0])
    head_dim = int(cos.shape[1]) * 2
    num_heads = int(d_model // head_dim) if head_dim > 0 else 1

    w1_key = next(k for k in state.keys() if k.endswith("ffn.w1.W"))
    d_ff = int(state[w1_key].shape[0])

    return {
        "vocab_size": int(vocab_size),
        "context_length": int(context_length),
        "d_model": int(d_model),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "d_ff": int(d_ff),
    }


def _get_end_token_id(tokenizer: Tokenizer, end_token: str) -> int | None:
    end_bytes = end_token.encode("utf-8")
    b2i = getattr(tokenizer, "_bytes_to_id", None)
    if isinstance(b2i, dict) and end_bytes in b2i:
        return int(b2i[end_bytes])
    return None


def _sample_next_token(
    logits: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_p >= 1.0:
        return torch.multinomial(probs, num_samples=1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    keep = cum <= top_p
    keep[0] = True  

    filtered = sorted_probs * keep
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    sampled_pos = torch.multinomial(filtered, num_samples=1) 
    return sorted_idx.gather(dim=-1, index=sampled_pos)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a trained Transformer LM checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    p.add_argument("--vocab_json", type=str, required=True, help="Path to vocab json produced by BPE training")
    p.add_argument("--merges_txt", type=str, required=True, help="Path to merges txt produced by BPE training")
    p.add_argument("--prompt", type=str, default="", help="Prompt text to condition on")
    p.add_argument("--prompt_path", type=str, default=None, help="Optional path to a text file containing the prompt")
    p.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    p.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (0 = greedy)")
    p.add_argument("--top_p", type=float, default=1.0, help="Top-p / nucleus sampling threshold (1.0 = disabled)")
    p.add_argument("--end_token", type=str, default="<|endoftext|>", help="Stop generation when this token is produced (if in vocab)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device

    prompt = args.prompt
    if args.prompt_path is not None:
        prompt = Path(args.prompt_path).read_text(encoding="utf-8")

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_json,
        merges_filepath=args.merges_txt,
        special_tokens=[args.end_token],
    )
    end_id = _get_end_token_id(tokenizer, args.end_token)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    assert isinstance(state, dict), "Checkpoint must be a state_dict or a dict containing 'model'"

    h = _infer_hparams_from_state_dict(state)
    model = TransformerLM(**h).to(device)
    model.load_state_dict(state)
    model.eval()

    ids = tokenizer.encode(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            idx_cond = idx[:, -model.context_length :]
            logits = model(idx_cond)  
            next_logits = logits[0, -1, :]
            next_id = _sample_next_token(next_logits, temperature=args.temperature, top_p=args.top_p)
            idx = torch.cat([idx, next_id.view(1, 1)], dim=1)

            if end_id is not None and int(next_id.item()) == end_id:
                break

    out_text = tokenizer.decode(idx[0].tolist())
    print(out_text)


if __name__ == "__main__":
    main()

