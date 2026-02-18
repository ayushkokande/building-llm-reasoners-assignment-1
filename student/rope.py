from __future__ import annotations

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        
        angles = positions[:, None] * freqs[None, :]
        

        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Position indices of shape (..., seq_len) or (seq_len,)
            
        Returns:
            Rotated tensor of same shape as x
        """
        in_dtype = x.dtype

        cos_selected = self.cos[token_positions]
        sin_selected = self.sin[token_positions]

        x = x.float()
        x_reshaped = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]

        x_even_rotated = x_even * cos_selected - x_odd * sin_selected
        x_odd_rotated = x_even * sin_selected + x_odd * cos_selected

        x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)
        return x_rotated.reshape(x.shape).to(in_dtype)
