from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init


class Linear(nn.Module):
    """
    Linear transformation module without bias.
    
    Performs: output = input @ W^T
    where W is stored as (out_features, in_features) for memory efficiency.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        std = (2.0 / (in_features + out_features)) ** 0.5
        init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation: output = x @ W^T
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        return x @ self.W.T
