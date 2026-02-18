from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, BinaryIO, Dict, IO, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from student.adamw import AdamW
from student.embedding import Embedding
from student.linear import Linear
from student.pretokenization_example import find_chunk_boundaries
from student.regexsplitter import RegexSplitter
from student.rmsnorm import RMSNorm
from student.rope import RotaryPositionalEmbedding
from student.swiglu import SwiGLU
from student.tokenizer import PAT as GPT2_PRETOKENIZE_PATTERN
from student.tokenizer import Tokenizer


def run_linear(
    d_in: int,
    d_out: int,
    weights: torch.Tensor,
    in_features: torch.Tensor,
) -> torch.Tensor:
    linear = Linear(in_features=d_in, out_features=d_out)
    linear.load_state_dict({"W": weights})
    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    emb = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    emb.load_state_dict({"weight": weights})
    return emb(token_ids)


def run_silu(in_features: torch.Tensor) -> torch.Tensor:
    return F.silu(in_features)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
    in_features: torch.Tensor,
) -> torch.Tensor:
    swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=in_features.device, dtype=w1_weight.dtype)
    swiglu.load_state_dict(
        {
            "w1.W": w1_weight.to(device=in_features.device),
            "w2.W": w2_weight.to(device=in_features.device),
            "w3.W": w3_weight.to(device=in_features.device),
        }
    )
    return swiglu(in_features)


def _softmax_stable(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    exp = torch.exp(x - x_max)
    return exp / exp.sum(dim=dim, keepdim=True)


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax over `dim`, where mask==False entries get probability 0.
    If a row is fully masked, returns all zeros for that row.
    """
    mask = mask.to(dtype=torch.bool, device=scores.device)
    scores = scores.masked_fill(~mask, float("-inf"))

    row_max = scores.max(dim=dim, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

    exp = torch.exp(scores - row_max) * mask.to(dtype=scores.dtype)
    denom = exp.sum(dim=dim, keepdim=True)
    return torch.where(denom > 0, exp / denom, torch.zeros_like(exp))


def run_softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    return _softmax_stable(in_features, dim=dim)


def run_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is None:
        probs = _softmax_stable(scores, dim=-1)
    else:
        probs = _masked_softmax(scores, mask=mask, dim=-1)
    return probs @ V


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    in_features: torch.Tensor,
) -> torch.Tensor:
    _ = d_model
    *batch_dims, seq_len, _d_in = in_features.shape
    if _d_in % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    head_dim = _d_in // num_heads

    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    Q = Q.view(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)
    K = K.view(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)

    causal = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool))
    ctx = run_scaled_dot_product_attention(Q=Q, K=K, V=V, mask=causal)

    ctx = ctx.transpose(-3, -2).reshape(*batch_dims, seq_len, _d_in)
    return ctx @ o_proj_weight.T


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_query_or_key.device)
    *batch_dims, seq_len, _ = in_query_or_key.shape
    if token_positions.shape[-1] != seq_len:
        raise ValueError("token_positions must have last dimension == seq_len")

    tp = token_positions.to(device=in_query_or_key.device)
    tp_batch = list(tp.shape[:-1])
    if len(tp_batch) > len(batch_dims):
        raise ValueError("token_positions has too many batch dimensions")
    if len(tp_batch) < len(batch_dims):
        tp = tp.reshape(*tp_batch, *([1] * (len(batch_dims) - len(tp_batch))), seq_len)
    tp = tp.expand(*batch_dims, seq_len)

    return rope(in_query_or_key, tp)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    in_features: torch.Tensor,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    _ = d_model
    *batch_dims, seq_len, _d_in = in_features.shape
    if _d_in % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    head_dim = _d_in // num_heads

    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    Q = Q.view(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)
    K = K.view(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)

    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device, dtype=torch.long)

    Q = run_rope(d_k=head_dim, theta=theta, max_seq_len=max_seq_len, in_query_or_key=Q, token_positions=token_positions)
    K = run_rope(d_k=head_dim, theta=theta, max_seq_len=max_seq_len, in_query_or_key=K, token_positions=token_positions)

    causal = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool))
    ctx = run_scaled_dot_product_attention(Q=Q, K=K, V=V, mask=causal)

    ctx = ctx.transpose(-3, -2).reshape(*batch_dims, seq_len, _d_in)
    return ctx @ o_proj_weight.T


def run_rmsnorm(d_model: int, eps: float, weights: torch.Tensor, in_features: torch.Tensor) -> torch.Tensor:
    norm = RMSNorm(d_model=d_model, eps=eps, device=in_features.device, dtype=weights.dtype)
    norm.load_state_dict({"weight": weights.to(device=in_features.device)})
    return norm(in_features)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, torch.Tensor],
    in_features: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-5
    x = in_features

    x1 = run_rmsnorm(d_model=d_model, eps=eps, weights=weights["ln1.weight"], in_features=x)
    attn_out = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=x1,
        token_positions=torch.arange(x.shape[-2], device=x.device, dtype=torch.long),
    )
    x = x + attn_out

    x2 = run_rmsnorm(d_model=d_model, eps=eps, weights=weights["ln2.weight"], in_features=x)
    ffn_out = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=x2,
    )
    x = x + ffn_out
    return x


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, torch.Tensor],
    in_indices: torch.Tensor,
) -> torch.Tensor:
    _ = vocab_size
    eps = 1e-5

    x = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights["token_embeddings.weight"],
        token_ids=in_indices,
    )

    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        layer_weights = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x,
        )

    x = run_rmsnorm(d_model=d_model, eps=eps, weights=weights["ln_final.weight"], in_features=x)
    logits = x @ weights["lm_head.weight"].T
    return logits


def run_get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(dataset.shape[0])
    if n <= context_length:
        raise ValueError("Dataset too small for requested context_length")
    start = np.random.randint(0, n - context_length, size=(batch_size,))
    x = np.stack([dataset[i : i + context_length] for i in start], axis=0)
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in start], axis=0)
    x_t = torch.tensor(x, dtype=torch.long, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    return x_t, y_t


def run_cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)
    nll = -log_probs.gather(dim=-1, index=targets.view(-1, 1)).squeeze(-1)
    return nll.mean()


def run_gradient_clipping(parameters: Any, max_l2_norm: float) -> None:
    eps = 1e-6
    grads = []
    for p in parameters:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        grads.append(g.detach())
    if not grads:
        return
    device = grads[0].device
    total_sq = torch.zeros((), device=device)
    for g in grads:
        total_sq = total_sq + (g.float().pow(2).sum())
    total_norm = torch.sqrt(total_sq)
    if total_norm <= max_l2_norm:
        return
    scale = float(max_l2_norm) / float(total_norm + eps)
    for p in parameters:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        g.mul_(scale)


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it <= warmup_iters:
        return max_learning_rate * (it / warmup_iters) if warmup_iters > 0 else max_learning_rate
    if it <= cosine_cycle_iters:
        t = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1.0 + math.cos(math.pi * t)) * (max_learning_rate - min_learning_rate)
    return min_learning_rate


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | Path | BinaryIO | IO[bytes],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": int(iteration),
        },
        out,
    )


def run_load_checkpoint(
    src: str | Path | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(src, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["iteration"])


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    return Tokenizer(vocab=dict(vocab), merges=merges, special_tokens=special_tokens)


def pre_tokenize(
    splitter: RegexSplitter,
    filepath: str,
    num_processes: int = 1,
    special_token: str = "<|endoftext|>",
) -> dict[str, int]:
    handle: BinaryIO = Path(filepath).open("rb")
    boundaries = find_chunk_boundaries(handle, max(num_processes * 4, 1), special_token.encode("utf-8"))
    handle.close()

    args = [(filepath, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    pre_token_counts: dict[str, int] = {}

    with Pool(num_processes) as p:
        results = p.starmap(splitter.seek_and_split, args)
    for pre_token_counts_sample in results:
        for k, v in pre_token_counts_sample.items():
            pre_token_counts[k] = pre_token_counts.get(k, 0) + v

    return pre_token_counts


Word = Tuple[int, ...]
Pair = Tuple[int, int]


def _iter_pairs(word: Word):
    for i in range(len(word) - 1):
        yield (word[i], word[i + 1])


def build_pair_indexes(word_freqs: Dict[Word, int]):
    pair_counts: Counter[Pair] = Counter()
    pair_to_words: Dict[Pair, set[Word]] = defaultdict(set)

    for w, freq in word_freqs.items():
        if len(w) < 2:
            continue
        for p in _iter_pairs(w):
            pair_counts[p] += freq
            pair_to_words[p].add(w)

    return pair_counts, pair_to_words


def _merge_in_word(word: Word, pair: Pair, new_id: int) -> Word:
    a, b = pair
    out: list[int] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def merge_pair_incremental(
    word_freqs: Dict[Word, int],
    pair_counts: Counter[Pair],
    pair_to_words: Dict[Pair, set[Word]],
    pair: Pair,
    new_id: int,
) -> None:
    affected = pair_to_words.get(pair)
    affected_words = list(affected)

    for w in affected_words:
        freq = word_freqs[w]
        if len(w) >= 2:
            for p in _iter_pairs(w):
                pair_counts[p] -= freq
                s = pair_to_words.get(p)
                if s is not None:
                    s.discard(w)
                    if not s:
                        del pair_to_words[p]
                if pair_counts[p] == 0:
                    del pair_counts[p]

        new_w = _merge_in_word(w, pair, new_id)

        del word_freqs[w]
        word_freqs[new_w] = word_freqs.get(new_w, 0) + freq

        if len(new_w) >= 2:
            for p in _iter_pairs(new_w):
                pair_counts[p] += freq
                pair_to_words.setdefault(p, set()).add(new_w)

    pair_counts.pop(pair, None)


def pretokenize(
    filepath: str | os.PathLike,
    special_tokens: list[str],
    pat: str,
    num_processes: int = 1,
) -> Counter[tuple[bytes, ...]]:
    splitter = RegexSplitter(pat=pat, special_tokens=special_tokens)

    pre_token_counts = pre_tokenize(
        splitter=splitter,
        filepath=str(filepath),
        num_processes=num_processes,
        special_token=special_tokens[0] if special_tokens else "<|endoftext|>",
    )
    counts: Counter[tuple[bytes, ...]] = Counter()
    for s, cnt in pre_token_counts.items():
        key = tuple(bytes([b]) for b in s.encode("utf-8"))
        counts[key] += cnt

    return counts


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    input_file_path = Path(input_path)

    vocab: dict[int, bytes] = {}
    next_id = 0
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    num_processes = kwargs.get("num_processes", 8)

    word_freqs = pretokenize(
        filepath=input_file_path,
        special_tokens=special_tokens,
        pat=GPT2_PRETOKENIZE_PATTERN,
        num_processes=num_processes,
    )

    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    pair_counts, pair_to_words = build_pair_indexes(word_freqs)
    num_merges = vocab_size - len(vocab)
    merges: list[tuple[bytes, bytes]] = []

    for _ in range(num_merges):
        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        a, b = best_pair
        merges.append((a, b))

        new_token = a + b
        vocab[next_id] = new_token
        next_id += 1

        merge_pair_incremental(word_freqs, pair_counts, pair_to_words, best_pair, new_token)

    return vocab, merges

