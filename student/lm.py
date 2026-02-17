from __future__ import annotations

import torch
import torch.nn as nn

from student.assignment1 import run_scaled_dot_product_attention
from student.embedding import Embedding
from student.linear import Linear
from student.rmsnorm import RMSNorm
from student.rope import RotaryPositionalEmbedding
from student.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with RoPE and SwiGLU FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len

        self.ln1 = RMSNorm(d_model, eps=eps)
        self.attn_q_proj = Linear(d_model, d_model)
        self.attn_k_proj = Linear(d_model, d_model)
        self.attn_v_proj = Linear(d_model, d_model)
        self.attn_output_proj = Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=self.head_dim, max_seq_len=max_seq_len)
        self.ln2 = RMSNorm(d_model, eps=eps)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        x1 = self.ln1(x)
        Q = self.attn_q_proj(x1).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.attn_k_proj(x1).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.attn_v_proj(x1).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        pos = torch.arange(T, device=x.device, dtype=torch.long)
        Q = self.rope(Q, pos)
        K = self.rope(K, pos)

        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        ctx = run_scaled_dot_product_attention(Q=Q, K=K, V=V, mask=causal)
        ctx = ctx.transpose(1, 2).reshape(B, T, C)

        x = x + self.attn_output_proj(ctx)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Transformer language model: embedding + N blocks + final norm + lm_head."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, eps=eps)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)

