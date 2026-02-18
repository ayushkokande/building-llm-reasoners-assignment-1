from __future__ import annotations

import torch
import torch.nn as nn

from student.linear import Linear


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network: SwiGLU(x) = (SiLU(x @ W1) * (x @ W3)) @ W2
    
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

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
        """
        up = self.w1(x)
        
        gate = self.w3(x)
        
        activated = up * torch.sigmoid(up) * gate
        
        output = self.w2(activated)
        
        return output
