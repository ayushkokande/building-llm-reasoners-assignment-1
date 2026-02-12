from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init


class Embedding(nn.Module):
    """
    Simple embedding layer (lookup table) without using nn.Embedding.

    - num_embeddings: vocabulary size
    - embedding_dim:  embedding size (d_model), last dimension of the weight matrix

    Weight shape: (num_embeddings, embedding_dim)
    Forward: weight[token_ids]
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Store embedding matrix with d_model (embedding_dim) as the final dimension.
        # Shape: (vocab_size, d_model)
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        # Initialize weights with truncated normal, similar to the Linear layer.
        init.trunc_normal_(self.weight, mean=0.0, std=0.02, a=-2 * 0.02, b=2 * 0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embeddings for given token IDs.

        Args:
            token_ids: integer tensor of any shape (...), containing token indices.

        Returns:
            Tensor of shape (..., embedding_dim) with the corresponding embeddings.
        """
        return self.weight[token_ids]

