from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    Normalizes input by RMS across the last dimension (d_model),
    then scales by a learnable weight parameter.
    
    Formula: output = (x / rms) * weight
    where rms = sqrt(mean(x^2) + eps)
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable scaling weight parameter, shape (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
               or any shape ending with d_model
            
        Returns:
            Normalized tensor of the same shape as x
        """
        # Store original dtype for downcasting later
        original_dtype = x.dtype
        
        # Upcast to float32 for numerical stability
        x_f32 = x.to(torch.float32) #convert the input tensor to float32 for numerical stability, in easy terms we are converting the input tensor to a float32 tensor. 
        # why we are doing this? because the input tensor is in float16 or float32, and we need to convert it to float32 for numerical stability.
        #To explain like you're 5: The input tensor is in float16 or float32, and we need to convert it to float32 for numerical stability.
        
        # Compute RMS: sqrt(mean(x^2) + eps)
        # mean over last dimension, keep dims for broadcasting
        x_squared = x_f32 ** 2
        mean_x_squared = x_squared.mean(dim=-1, keepdim=True)#taking dim=-1 means that we are taking the mean of the last dimension, which is the d_model dimension.
        # keepdim=True means that we are keeping the dimension of the mean_x_squared tensor.
        # why we are doing this? because we need to keep the dimension of the mean_x_squared tensor to be the same as the input tensor.
        #To explain like you're 5: The mean_x_squared tensor is in float32, and we need to keep the dimension of the mean_x_squared tensor to be the same as the input tensor.
        rms = torch.sqrt(mean_x_squared + self.eps)
        
        # Normalize: x / rms
        x_norm = x_f32 / rms
        
        # Scale by learnable weight
        # weight is (d_model,), so it broadcasts correctly
        output = x_norm * self.weight
        
        # Downcast back to original dtype
        return output.to(original_dtype)
