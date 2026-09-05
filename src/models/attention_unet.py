"""Attention U-Net implementation for the reproduction study."""

import torch
from torch import nn

from .unet import DoubleConv


class AttentionGate(nn.Module):
    """Attention gate for a skip feature map conditioned on decoder features."""

    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.theta = nn.Conv2d(skip_channels, inter_channels, 2, stride=2, bias=False)
        self.phi = nn.Conv2d(gate_channels, inter_channels, 1, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, 1, bias=True)
        self.norm = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        if theta_x.shape[-2:] != phi_g.shape[-2:]:
            phi_g = nn.functional.interpolate(phi_g, size=theta_x.shape[-2:], mode="bilinear", align_corners=False)
        psi = self.sigmoid(self.psi(self.relu(theta_x + phi_g)))
        psi = nn.functional.interpolate(psi, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * psi


class AttentionUNet(nn.Module):
    """U-Net variant with attention gates on skip connections."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.att4 = AttentionGate(c * 8, c * 8, c * 4)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.att3 = AttentionGate(c * 4, c * 4, c * 2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.att2 = AttentionGate(c * 2, c * 2, c)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.att1 = AttentionGate(c, c, max(c // 2, 1))
        self.dec1 = DoubleConv(c * 2, c)
        self.out = nn.Conv2d(c, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4 = self.att4(e4, d4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        e3 = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        e2 = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        e1 = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)
