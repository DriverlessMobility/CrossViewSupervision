from typing import Optional

import torch
import torch.nn as nn

from mmcv.utils import Registry, build_from_cfg

AID_ADAPTERS = Registry('aid_adapters')
AID_DOWNS = Registry('aid_downs')
AID_FUSERS = Registry('aid_fusers')


@AID_ADAPTERS.register_module()
class AffineAdapter(nn.Module):
    def __init__(self, channels: int, height: int = 1, width: int = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, height, width))
        self.beta = nn.Parameter(torch.zeros(1, channels, height, width))

    def forward(self, x):
        return x * self.gamma + self.beta


@AID_DOWNS.register_module()
class DownsampleCNN(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=128, use_gn=True):
        super().__init__()
        n1 = nn.GroupNorm(min(32, hidden_dim), hidden_dim) if use_gn else nn.Identity()
        n2 = nn.GroupNorm(min(32, hidden_dim * 2), hidden_dim * 2) if use_gn else nn.Identity()
        if use_gn:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 3, stride=2, padding=1, bias=False),
                n1, nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1, bias=False),
                n2, nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


@AID_FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        self.in_channels = in_ch
        self.out_channels = out_ch
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs)


def build_aid_fuser(fusion_mode: str, bev_embed_dims: int, fuser_cfg: Optional[dict]):
    if fusion_mode != 'fuse':
        return nn.Identity()
    cfg = fuser_cfg or dict(type='ConvFuser', in_ch=2 * bev_embed_dims, out_ch=bev_embed_dims)
    return build_from_cfg(cfg, AID_FUSERS)
