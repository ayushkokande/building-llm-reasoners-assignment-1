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
        in_features: int, #final dimension of the input tensor
        out_features: int, #final dimension of the output tensor
        device: torch.device | None = None, #device to store the tensor on
        dtype: torch.dtype | None = None, #datatype of the parameters.
    ) -> None:
        super().__init__()
        self.in_features = in_features #final dimension of the input tensor
        self.out_features = out_features #final dimension of the output tensor

        # Create weight parameter W with shape (out_features, in_features)
        # This is stored as W (not W^T) for memory ordering reasons
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype)) #This creates W with shape (out_features, in_features)
        
        # Initialize weights using truncated normal distribution
        init.trunc_normal_(self.W, mean=0.0, std=0.02, a=-2 * 0.02, b=2 * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation: output = x @ W^T
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Since W is (out_features, in_features), we transpose it to get W^T
        # Then compute: x @ W^T
        return x @ self.W.T
