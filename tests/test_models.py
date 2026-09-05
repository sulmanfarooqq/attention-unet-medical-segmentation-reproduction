import torch

from src.models.attention_unet import AttentionUNet
from src.models.unet import UNet


def test_unet_shape() -> None:
    model = UNet(in_channels=1, out_channels=1, base_channels=8)
    x = torch.randn(2, 1, 128, 128)
    assert model(x).shape == x.shape


def test_attention_unet_shape() -> None:
    model = AttentionUNet(in_channels=1, out_channels=1, base_channels=8)
    x = torch.randn(2, 1, 128, 128)
    assert model(x).shape == x.shape
