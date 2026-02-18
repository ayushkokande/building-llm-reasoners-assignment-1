from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    exp = torch.exp(x - x_max)
    return exp / exp.sum(dim=dim, keepdim=True)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax over `dim`, where mask==False entries get probability 0.
    If a row is fully masked, returns all zeros for that row.
    """
    mask = mask.to(dtype=torch.bool, device=scores.device)
    scores = scores.masked_fill(~mask, float("-inf"))

    row_max = scores.max(dim=dim, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

    exp = torch.exp(scores - row_max) * mask.to(dtype=scores.dtype)
    denom = exp.sum(dim=dim, keepdim=True)
    return torch.where(denom > 0, exp / denom, torch.zeros_like(exp))
