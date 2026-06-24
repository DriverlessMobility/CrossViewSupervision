from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from mmcv.utils import Registry, build_from_cfg

from plugin.models.aid.aid_modules import AID_ADAPTERS, AID_DOWNS, build_aid_fuser
from plugin.models.aid.encoders import AID_ENCODERS
from plugin.models.aid.utils.tensor_ops import bev_norm, center_crop

AID_PIPELINES = Registry('aid_pipelines')


def _ensure_fusion_mode(fusion_mode: str) -> str:
    if fusion_mode not in ('ego', 'aid', 'fuse'):
        raise ValueError(f"Unsupported fusion_mode '{fusion_mode}'.")
    return fusion_mode


def _build_loss(loss_name: str):
    name = loss_name.lower()
    if name == 'mse':
        return nn.functional.mse_loss

    raise ValueError(f"loss_bev_function '{loss_name}' not implemented.")


@AID_PIPELINES.register_module()
class AIDFusionPipeline(nn.Module):
    """Online AID4AD pipeline with configurable fusion/crop options."""

    def __init__(
        self,
        bev_embed_dims: int,
        encoder: dict,
        downsampler: Optional[dict] = None,
        fuser: Optional[dict] = None,
        fusion_mode: str = 'fuse',
        crop_size: Optional[Tuple[int, int]] = None,
        **_,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.fusion_mode = _ensure_fusion_mode(fusion_mode)

        self.encoder = build_from_cfg(encoder, AID_ENCODERS)
        encoder_out_ch = getattr(self.encoder, 'out_channels', 64)

        if downsampler is None:
            self.down = nn.Identity()
        else:
            cfg = dict(downsampler)
            cfg['in_channels'] = encoder_out_ch
            self.down = build_from_cfg(cfg, AID_DOWNS)

        self.fuser = build_aid_fuser(self.fusion_mode, bev_embed_dims, fuser)

    def forward(
        self,
        bev_feats: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
        cond: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        aid = self._encode(inputs)
        if self.crop_size is not None:
            aid = center_crop(aid, self.crop_size)
        aid = self.down(aid)
        out = self._fuse(bev_feats, aid)
        return (out, {}) if return_aux else out

    def _encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if 'aerial_img' not in inputs:
            raise KeyError("AIDFusionPipeline expects 'aerial_img' in inputs.")
        return self.encoder(inputs['aerial_img'])

    def _fuse(self, bev_feats: torch.Tensor, aid_feats: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == 'ego':
            return bev_feats
        if self.fusion_mode == 'aid':
            return aid_feats
        return self.fuser(torch.cat([bev_feats, aid_feats], dim=1))


@AID_PIPELINES.register_module()
class CrossViewSupervisionPipeline(nn.Module):
    """Cross-view supervision via aerial encoder (teacher loss, ego BEV output)."""

    def __init__(
        self,
        encoder: dict,
        downsampler: Optional[dict] = None,
        adapter_bev_teaching: Optional[dict] = None,
        loss_bev_weight: float = 1.0,
        loss_bev_function: str = 'MSE',
        loss_bev_norm: bool = False,
        fuser: Optional[dict] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        **_,
    ) -> None:
        super().__init__()
        if loss_bev_weight <= 0.0:
            raise ValueError("CrossViewSupervisionPipeline requires loss_bev_weight > 0.")
        self.loss_bev_weight = float(loss_bev_weight)
        self.loss_fn = _build_loss(loss_bev_function)
        self.loss_bev_norm = loss_bev_norm
        self.crop_size = crop_size

        self.encoder = build_from_cfg(encoder, AID_ENCODERS)
        encoder_out_ch = getattr(self.encoder, 'out_channels', 64)
        if downsampler is None:
            self.down = nn.Identity()
        else:
            cfg = dict(downsampler)
            cfg['in_channels'] = encoder_out_ch
            self.down = build_from_cfg(cfg, AID_DOWNS)

        self.adapter_bev_teaching = (
            build_from_cfg(adapter_bev_teaching, AID_ADAPTERS)
            if adapter_bev_teaching is not None
            else nn.Identity()
        )
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        for module in (self.encoder, self.down):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        bev_feats: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
        cond: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        aid = self.encoder(inputs['aerial_img'])
        if self.crop_size is not None:
            aid = center_crop(aid, self.crop_size)
        aid = self.down(aid)
        if return_aux:
            aux = {'loss_bev': self._compute_loss(bev_feats, aid)}
            return bev_feats, aux
        return bev_feats

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_teacher()
        return self

    def _compute_loss(self, bev_feats: torch.Tensor, aid_feats: torch.Tensor) -> torch.Tensor:
        aid_t = aid_feats.detach()
        bev_t = bev_feats
        if self.loss_bev_norm:
            aid_t = bev_norm(aid_t)
            bev_t = bev_norm(bev_t)
        bev_t = self.adapter_bev_teaching(bev_t)
        return self.loss_fn(bev_t, aid_t, reduction='mean') * self.loss_bev_weight
