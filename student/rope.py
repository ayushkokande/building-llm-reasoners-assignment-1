from __future__ import annotations

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    
    Applies rotary rotations to pairs of dimensions based on token positions.
    For position m and dimension pair (2i, 2i+1), rotation angle is m * theta^(-2i/d_k).
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

        # Precompute frequencies: theta^(-2i/d_k) for i in [0, d_k//2)
        # Shape: (d_k // 2,)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        
        # Precompute positions: [0, 1, 2, ..., max_seq_len-1]
        # Shape: (max_seq_len,)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        
        # Outer product: positions[:, None] * freqs[None, :]
        # Shape: (max_seq_len, d_k // 2)
        angles = positions[:, None] * freqs[None, :]
        
        # Precompute cos and sin buffers
        # Shape: (max_seq_len, d_k // 2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # Register as buffers (not parameters, so they don't require gradients)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Position indices of shape (..., seq_len) or (seq_len,)
            
        Returns:
            Rotated tensor of same shape as x
        """
        # Get the shape info
        *batch_dims, seq_len, d_k = x.shape
        
        # Ensure token_positions has the right shape for broadcasting
        # If it's (seq_len,), expand to match batch dimensions by repeating
        if token_positions.ndim == 1:
            # Expand to (*batch_dims, seq_len) by adding batch dims and broadcasting
            # First add leading dims: (seq_len,) -> (1, ..., 1, seq_len)
            token_positions = token_positions.view(*([1] * len(batch_dims)), seq_len)
            # Then expand to match batch: (1, ..., 1, seq_len) -> (*batch_dims, seq_len)
            token_positions = token_positions.expand(*batch_dims, seq_len)
        
        # Reshape x to separate pairs: (..., seq_len, d_k//2, 2)
        x_reshaped = x.reshape(*batch_dims, seq_len, d_k // 2, 2)
        
        # Extract pairs: x_even = x[..., 0], x_odd = x[..., 1]
        # Shape: (..., seq_len, d_k//2)
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]
        
        # Index cos and sin using token_positions
        # token_positions: (*batch_dims, seq_len) -> flatten to get indices
        pos_flat = token_positions.flatten().long()
        
        # Index cos/sin: self.cos[pos_flat] -> (N, d_k//2) where N = prod(batch_dims) * seq_len
        cos_selected = self.cos[pos_flat]
        sin_selected = self.sin[pos_flat]
        
        # Reshape back to match x_even/x_odd: (*batch_dims, seq_len, d_k//2)
        cos_selected = cos_selected.reshape(*batch_dims, seq_len, d_k // 2)
        sin_selected = sin_selected.reshape(*batch_dims, seq_len, d_k // 2)
        
        # Apply rotation:
        # x_even' = x_even * cos - x_odd * sin
        # x_odd' = x_even * sin + x_odd * cos
        x_even_rotated = x_even * cos_selected - x_odd * sin_selected
        x_odd_rotated = x_even * sin_selected + x_odd * cos_selected
        
        # Stack back: (..., seq_len, d_k//2, 2)
        x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)
        
        # Reshape back to original: (..., seq_len, d_k)
        return x_rotated.reshape(*batch_dims, seq_len, d_k)
