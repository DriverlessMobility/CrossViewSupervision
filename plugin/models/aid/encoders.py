import torch.nn as nn

from mmcv.utils import Registry

AID_ENCODERS = Registry('aid_encoders')


@AID_ENCODERS.register_module()
class ResNetUNet(nn.Module):
    """Lightweight ResNet-based UNet encoder used for AID fusion and CrossView."""

    def __init__(self, outC: int = 64):
        super().__init__()
        from .resunet import ResNetUNet as _ResNetUNet
        self.net = _ResNetUNet(outC=outC)
        self.out_channels = outC

    def forward(self, x):
        return self.net(x)
