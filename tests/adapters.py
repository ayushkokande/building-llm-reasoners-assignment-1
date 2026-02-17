# pyright: ignore
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

# All substantive logic lives in student/.
from student import assignment1 as impl


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],  # type: ignore
    in_features: Float[Tensor, " ... d_in"],  # type: ignore
) -> Float[Tensor, " ... d_out"]:  # type: ignore
    return impl.run_linear(d_in=d_in, d_out=d_out, weights=weights, in_features=in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, "vocab_size d_model"],  # type: ignore
    token_ids: Int[Tensor, " ..."],  # type: ignore
) -> Float[Tensor, " ... d_model"]:  # type: ignore
    return impl.run_embedding(vocab_size=vocab_size, d_model=d_model, weights=weights, token_ids=token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, "d_ff d_model"],  # type: ignore
    w2_weight: Float[Tensor, "d_model d_ff"],  # type: ignore
    w3_weight: Float[Tensor, "d_ff d_model"],  # type: ignore
    in_features: Float[Tensor, " ... d_model"],  # type: ignore
) -> Float[Tensor, " ... d_model"]:  # type: ignore
    return impl.run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=in_features,
    )


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],  # type: ignore
    K: Float[Tensor, " ... keys d_k"],  # type: ignore
    V: Float[Tensor, " ... values d_v"],  # type: ignore
    mask: Bool[Tensor, " ... queries keys"] | None = None,  # type: ignore
) -> Float[Tensor, " ... queries d_v"]:  # type: ignore
    return impl.run_scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, "d_k d_in"],  # type: ignore
    k_proj_weight: Float[Tensor, "d_k d_in"],  # type: ignore
    v_proj_weight: Float[Tensor, "d_v d_in"],  # type: ignore
    o_proj_weight: Float[Tensor, "d_model d_v"],  # type: ignore
    in_features: Float[Tensor, " ... sequence_length d_in"],  # type: ignore
) -> Float[Tensor, " ... sequence_length d_out"]:  # type: ignore
    return impl.run_multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_features,
    )


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, "d_k d_in"],  # type: ignore
    k_proj_weight: Float[Tensor, "d_k d_in"],  # type: ignore
    v_proj_weight: Float[Tensor, "d_v d_in"],  # type: ignore
    o_proj_weight: Float[Tensor, "d_model d_v"],  # type: ignore
    in_features: Float[Tensor, "... sequence_length d_in"],  # type: ignore
    token_positions: Int[Tensor, "... sequence_length"] | None = None,  # type: ignore
) -> Float[Tensor, "... sequence_length d_out"]:  # type: ignore
    return impl.run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_features,
        token_positions=token_positions,
    )


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],  # type: ignore
    token_positions: Int[Tensor, " ... sequence_length"],  # type: ignore
) -> Float[Tensor, " ... sequence_length d_k"]:  # type: ignore
    return impl.run_rope(
        d_k=d_k,
        theta=theta,
        max_seq_len=max_seq_len,
        in_query_or_key=in_query_or_key,
        token_positions=token_positions,
    )


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, "batch sequence_length d_model"],  # type: ignore
) -> Float[Tensor, "batch sequence_length d_model"]:  # type: ignore
    return impl.run_transformer_block(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        weights=weights,
        in_features=in_features,
    )


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, "batch_size sequence_length"],  # type: ignore
) -> Float[Tensor, "batch_size sequence_length vocab_size"]:  # type: ignore
    return impl.run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        weights=weights,
        in_indices=in_indices,
    )


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, "d_model"],  # type: ignore
    in_features: Float[Tensor, " ... d_model"],  # type: ignore
) -> Float[Tensor, " ... d_model"]:  # type: ignore
    return impl.run_rmsnorm(d_model=d_model, eps=eps, weights=weights, in_features=in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:  # type: ignore
    return impl.run_silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return impl.run_get_batch(dataset=dataset, batch_size=batch_size, context_length=context_length, device=device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:  # type: ignore
    return impl.run_softmax(in_features=in_features, dim=dim)


def run_cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]  # type: ignore
) -> Float[Tensor, ""]:  # type: ignore
    return impl.run_cross_entropy(inputs=inputs, targets=targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    return impl.run_gradient_clipping(parameters=parameters, max_l2_norm=max_l2_norm)


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    return impl.get_adamw_cls()


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    return impl.run_get_lr_cosine_schedule(
        it=it,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    return impl.run_save_checkpoint(model=model, optimizer=optimizer, iteration=iteration, out=out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    return impl.run_load_checkpoint(src=src, model=model, optimizer=optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return impl.get_tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return impl.run_train_bpe(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens, **kwargs)
