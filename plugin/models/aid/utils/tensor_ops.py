from typing import Tuple
import torch


def center_crop(x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """Crop tensor around its spatial center."""
    out_h, out_w = out_hw
    height, width = x.shape[-2:]
    y0 = (height - out_h) // 2
    x0 = (width - out_w) // 2
    return x[..., y0:y0 + out_h, x0:x0 + out_w]

def bev_norm(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-6)

