from __future__ import annotations

import torch
import torch.nn as nn

from student.linear import Linear


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network: SwiGLU(x) = (SiLU(x @ W1) * (x @ W3)) @ W2
    
    Where:
    - W1: up-project (d_ff, d_model)
    - W3: gate projection (d_ff, d_model)
    - W2: down-project (d_model, d_ff)
    - SiLU(x) = x * sigmoid(x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Three linear layers for SwiGLU
        # W1: up-project from d_model to d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # W3: gate projection from d_model to d_ff
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # W2: down-project from d_ff to d_model
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
        """
        # Up-project: x @ W1.T -> (..., d_ff)
        up = self.w1(x)
        
        # Gate projection: x @ W3.T -> (..., d_ff)
        gate = self.w3(x)
        
        # Apply SiLU to up and multiply by gate
        # SiLU(x) = x * sigmoid(x)
        activated = up * torch.sigmoid(up) * gate
        
        # Down-project: activated @ W2.T -> (..., d_model)
        output = self.w2(activated)
        
        return output
