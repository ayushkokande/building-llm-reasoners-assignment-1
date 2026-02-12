# pyright: ignore
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO, Dict, Tuple
from pathlib import Path

from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import regex as re

from student.pretokenization_example import find_chunk_boundaries
from student.tokenizer import Tokenizer
from student.regexsplitter import RegexSplitter
from student.linear import Linear


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],  # type: ignore
    in_features: Float[Tensor, " ... d_in"],  # type: ignore
) -> Float[Tensor, " ... d_out"]:  # type: ignore
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    
    linear = Linear(in_features=d_in, out_features=d_out, device=weights.device, dtype=weights.dtype)
    linear.load_state_dict({"W": weights})
    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, "vocab_size d_model"],  # type: ignore
    token_ids: Int[Tensor, " ..."],  # type: ignore
) -> Float[Tensor, " ... d_model"]:  # type: ignore
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, "d_ff d_model"],  # type: ignore
    w2_weight: Float[Tensor, "d_model d_ff"],  # type: ignore
    w3_weight: Float[Tensor, "d_ff d_model"],  # type: ignore
    in_features: Float[Tensor, " ... d_model"],  # type: ignore
) -> Float[Tensor, " ... d_model"]:  # type: ignore
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],  # type: ignore
    K: Float[Tensor, " ... keys d_k"],  # type: ignore
    V: Float[Tensor, " ... values d_v"],  # type: ignore
    mask: Bool[Tensor, " ... queries keys"] | None = None,  # type: ignore
) -> Float[Tensor, " ... queries d_v"]:  # type: ignore
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, "d_k d_in"],  # type: ignore
    k_proj_weight: Float[Tensor, "d_k d_in"],  # type: ignore
    v_proj_weight: Float[Tensor, "d_v d_in"],  # type: ignore
    o_proj_weight: Float[Tensor, "d_model d_v"],  # type: ignore
    in_features: Float[Tensor, " ... sequence_length d_in"],  # type: ignore
) -> Float[Tensor, " ... sequence_length d_out"]:  # type: ignore
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Note: the d_k and d_v here are really d_k * num_heads and d_v * num_heads. See run_transformer_block for more documentation.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


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
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Note: the d_k and d_v here are really d_k * num_heads and d_v * num_heads. See run_transformer_block for more documentation.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],  # type: ignore
    token_positions: Int[Tensor, " ... sequence_length"],  # type: ignore
) -> Float[Tensor, " ... sequence_length d_k"]:  # type: ignore
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, "batch sequence_length d_model"],  # type: ignore
) -> Float[Tensor, "batch sequence_length d_model"]:  # type: ignore
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


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
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, "d_model"],  # type: ignore
    in_features: Float[Tensor, " ... d_model"],  # type: ignore
) -> Float[Tensor, " ... d_model"]:  # type: ignore
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:  # type: ignore
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:  # type: ignore
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]  # type: ignore
) -> Float[Tensor, ""]:  # type: ignore
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


#Returns pretoken frequencies for all of the chunks in the file
def pre_tokenize(
    splitter: RegexSplitter,
    filepath: str,
    num_processes: int = 1,
    special_token: str = "<|endoftext|>",
) -> dict[str, int]:
    """
    Run pre-tokenization algorithm using chunk boundaries.

    1. Splits file by byte ranges separated by the given `special_token`
    2. Each chunk is processed by `splitter.seek_and_split` (optionally in parallel)
    """
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
    pair_counts: Counter[Pair] = Counter() #counts of each pair
    pair_to_words: Dict[Pair,set[Word]] = defaultdict(set) #pair to words mapping is basically words which contain the pair.

    for w, freq in word_freqs.items():
        if len(w) < 2:
            continue #skip words less than 2 bytes
        for p in _iter_pairs(w):
            pair_counts[p] += freq #increment the count of the pair
            pair_to_words[p].add(w) #add the word to the pair to words mapping

    return pair_counts, pair_to_words


def _merge_in_word(word: Word, pair: Pair, new_id: int) -> Word:
    a, b = pair
    out: list[int] =  [] 
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)   # NOTE: new_token is bytes (can be multi-byte), not "a single byte"
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
    """
    In-place update:
      - word_freqs
      - pair_counts
      - pair_to_words
    for merging `pair` -> `new_token`.

    Only updates words that actually contain `pair` (via pair_to_words[pair]).
    """
    affected = pair_to_words.get(pair) #get the words that contain the pair

    # Copy because we will mutate maps/sets as we go
    affected_words = list(affected) #convert the set to a list

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
    """
    Pretokenize the file by using `find_chunk_boundaries` + `pre_tokenize`,
    then convert pre-token strings to tuples of single-byte `bytes` as expected
    by the BPE training loop.
    """
    splitter = RegexSplitter(pat=pat, special_tokens=special_tokens)

    pre_token_counts = pre_tokenize(
        splitter=splitter,
        filepath=str(filepath),
        num_processes=num_processes,
        special_token=special_tokens[0] if special_tokens else "<|endoftext|>",
    )
    #converts pretoken string counts to a tuple of byte ids to count mappings word -> [(id1, id2, id3, id4)]
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
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.
    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
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

