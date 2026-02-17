from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from multiprocessing import Pool
from pathlib import Path
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import regex as re
import torch
from torch import Tensor

from student.adamw import AdamW
from student.embedding import Embedding
from student.linear import Linear
from student.pretokenization_example import find_chunk_boundaries
from student.regexsplitter import RegexSplitter
from student.rmsnorm import RMSNorm
from student.rope import RotaryPositionalEmbedding
from student.swiglu import SwiGLU
from student.tokenizer import Tokenizer

###############################################################################
# Core NN building blocks (used by tests via adapters)
###############################################################################


def run_linear(d_in: int, d_out: int, weights: Tensor, in_features: Tensor) -> Tensor:
    linear = Linear(in_features=d_in, out_features=d_out, device=weights.device, dtype=weights.dtype)
    linear.load_state_dict({"W": weights})
    return linear(in_features)


def run_embedding(vocab_size: int, d_model: int, weights: Tensor, token_ids: Tensor) -> Tensor:
    emb = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=weights.device, dtype=weights.dtype)
    emb.load_state_dict({"weight": weights})
    return emb(token_ids)


def run_swiglu(d_model: int, d_ff: int, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor, in_features: Tensor) -> Tensor:
    swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=w1_weight.device, dtype=w1_weight.dtype)
    swiglu.w1.load_state_dict({"W": w1_weight})
    swiglu.w2.load_state_dict({"W": w2_weight})
    swiglu.w3.load_state_dict({"W": w3_weight})
    return swiglu(in_features)


def run_scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    d_k = Q.shape[-1]
    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    if mask is not None:
        probs = probs.masked_fill(~mask, 0.0)
        denom = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(denom > 0, probs / denom, probs)
    return torch.matmul(probs, V)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    *batch_dims, seq_len, _ = in_features.shape

    d_k_total = q_proj_weight.shape[0]
    d_v_total = v_proj_weight.shape[0]

    assert d_k_total % num_heads == 0, "d_k_total must be divisible by num_heads"
    assert d_v_total % num_heads == 0, "d_v_total must be divisible by num_heads"

    head_dim_k = d_k_total // num_heads
    head_dim_v = d_v_total // num_heads

    qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    qkv = in_features @ qkv_weight.transpose(-1, -2)

    Q, K, V = torch.split(qkv, [d_k_total, d_k_total, d_v_total], dim=-1)

    Q = Q.reshape(*batch_dims, seq_len, num_heads, head_dim_k).transpose(-3, -2)
    K = K.reshape(*batch_dims, seq_len, num_heads, head_dim_k).transpose(-3, -2)
    V = V.reshape(*batch_dims, seq_len, num_heads, head_dim_v).transpose(-3, -2)

    scale = 1.0 / math.sqrt(head_dim_k)
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

    causal = torch.tril(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(probs, V)

    ctx = ctx.transpose(-3, -2).reshape(*batch_dims, seq_len, d_v_total)
    out = ctx @ o_proj_weight.transpose(-1, -2)
    return out


def run_rope(d_k: int, theta: float, max_seq_len: int, in_query_or_key: Tensor, token_positions: Tensor) -> Tensor:
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor | None = None,
) -> Tensor:
    *batch_dims, seq_len, _ = in_features.shape

    d_k_total = q_proj_weight.shape[0]
    d_v_total = v_proj_weight.shape[0]

    assert d_k_total % num_heads == 0
    assert d_v_total % num_heads == 0

    head_dim = d_k_total // num_heads
    head_dim_v = d_v_total // num_heads

    qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    qkv = in_features @ qkv_weight.transpose(-1, -2)

    Q, K, V = torch.split(qkv, [d_k_total, d_k_total, d_v_total], dim=-1)

    Q = Q.reshape(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)
    K = K.reshape(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)
    V = V.reshape(*batch_dims, seq_len, num_heads, head_dim_v).transpose(-3, -2)

    if token_positions is None:
        rope_pos = torch.arange(seq_len, device=in_features.device, dtype=torch.long)
    else:
        rope_pos = token_positions.squeeze(0) if token_positions.ndim > 1 and token_positions.shape[0] == 1 else token_positions

    Q = run_rope(head_dim, theta, max_seq_len, Q, rope_pos)
    K = run_rope(head_dim, theta, max_seq_len, K, rope_pos)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool))
    ctx = run_scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    ctx = ctx.transpose(-3, -2).reshape(*batch_dims, seq_len, d_v_total)
    out = ctx @ o_proj_weight.transpose(-1, -2)
    return out


def run_rmsnorm(d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    rmsnorm = RMSNorm(d_model=d_model, eps=eps, device=weights.device, dtype=weights.dtype)
    rmsnorm.load_state_dict({"weight": weights})
    return rmsnorm(in_features)


def run_silu(in_features: Tensor) -> Tensor:
    return in_features * torch.sigmoid(in_features)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    batch, seq_len, _ = in_features.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    head_dim = d_model // num_heads

    def _linear(x: Tensor, W: Tensor, d_in: int, d_out: int) -> Tensor:
        W_out_in = W if W.shape[1] == d_in else W.T
        return run_linear(d_in, d_out, W_out_in, x)

    x = in_features
    x1 = run_rmsnorm(d_model, 1e-5, weights["ln1.weight"], x)

    Wq = weights["attn.q_proj.weight"]
    Wk = weights["attn.k_proj.weight"]
    Wv = weights["attn.v_proj.weight"]
    Wo = weights["attn.output_proj.weight"]

    if Wq.shape == Wk.shape == Wv.shape and Wq.ndim == 2:
        if Wq.shape[1] == d_model:
            Wqkv = torch.cat([Wq, Wk, Wv], dim=0)
            qkv = x1 @ Wqkv.transpose(-1, -2)
        elif Wq.shape[0] == d_model:
            Wqkv = torch.cat([Wq, Wk, Wv], dim=1)
            qkv = x1 @ Wqkv
        else:
            Q = _linear(x1, Wq, d_model, d_model)
            K = _linear(x1, Wk, d_model, d_model)
            V = _linear(x1, Wv, d_model, d_model)
            qkv = torch.cat([Q, K, V], dim=-1)
    else:
        Q = _linear(x1, Wq, d_model, d_model)
        K = _linear(x1, Wk, d_model, d_model)
        V = _linear(x1, Wv, d_model, d_model)
        qkv = torch.cat([Q, K, V], dim=-1)

    Q, K, V = torch.split(qkv, [d_model, d_model, d_model], dim=-1)

    Q = Q.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    K = K.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    V = V.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
    Q = run_rope(head_dim, theta, max_seq_len, Q, token_positions)
    K = run_rope(head_dim, theta, max_seq_len, K, token_positions)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
    ctx = run_scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    ctx = ctx.transpose(1, 2).reshape(batch, seq_len, d_model)

    attn_out = _linear(ctx, Wo, d_model, d_model)
    y = x + attn_out

    y1 = run_rmsnorm(d_model, 1e-5, weights["ln2.weight"], y)

    W1 = weights["ffn.w1.weight"]
    W2 = weights["ffn.w2.weight"]
    W3 = weights["ffn.w3.weight"]

    ff_out = run_swiglu(d_model, d_ff, W1, W2, W3, y1)

    out = y + ff_out
    return out


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    x = run_embedding(vocab_size=vocab_size, d_model=d_model, weights=weights["token_embeddings.weight"], token_ids=in_indices)

    for i in range(num_layers):
        prefix = f"layers.{i}."
        block_weights = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=block_weights,
            in_features=x,
        )

    x = run_rmsnorm(d_model, 1e-5, weights["ln_final.weight"], x)
    logits = run_linear(d_in=d_model, d_out=vocab_size, weights=weights["lm_head.weight"], in_features=x)
    return logits


###############################################################################
# Training utilities (data, loss, optimizer, schedule, checkpointing)
###############################################################################


def run_get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    max_start = n - context_length
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])
    x_t = torch.tensor(x, dtype=torch.long, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    return x_t, y_t


def run_softmax(in_features: Tensor, dim: int) -> Tensor:
    x_max = in_features.max(dim=dim, keepdim=True)[0]
    x_shifted = in_features - x_max
    exp_x = torch.exp(x_shifted)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp


def run_cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    log_softmax = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)
    log_probs = log_softmax.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (-log_probs).mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef)


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    t = it
    alpha_max = max_learning_rate
    alpha_min = min_learning_rate
    T_w = warmup_iters
    T_c = cosine_cycle_iters

    if t < T_w:
        return (t / T_w) * alpha_max
    if t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        return alpha_min + 0.5 * (1 + math.cos(progress * math.pi)) * (alpha_max - alpha_min)
    return alpha_min


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


###############################################################################
# Tokenizer helpers + BPE training (used by tests via adapters)
###############################################################################


def get_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> Tokenizer:
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


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


Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


def _iter_pairs(word: Word):
    for i in range(len(word) - 1):
        yield (word[i], word[i + 1])


def build_pair_indexes(word_freqs: dict[Word, int]):
    pair_counts: Counter[Pair] = Counter()
    pair_to_words: dict[Pair, set[Word]] = defaultdict(set)

    for w, freq in word_freqs.items():
        if len(w) < 2:
            continue
        for p in _iter_pairs(w):
            pair_counts[p] += freq
            pair_to_words[p].add(w)

    return pair_counts, pair_to_words


def _merge_in_word(word: Word, pair: Pair, new_token: bytes) -> Word:
    a, b = pair
    out: list[bytes] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_token)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def merge_pair_incremental(
    word_freqs: dict[Word, int],
    pair_counts: Counter[Pair],
    pair_to_words: dict[Pair, set[Word]],
    pair: Pair,
    new_token: bytes,
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

        new_w = _merge_in_word(w, pair, new_token)

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


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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
        pat=PAT,
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

