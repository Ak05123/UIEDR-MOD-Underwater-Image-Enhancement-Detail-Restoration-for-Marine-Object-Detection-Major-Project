"""Residual CNN detail-restoration network (Phase 5).

A compact residual network that refines the classically-enhanced image before
feature extraction. Operates on normalised RGB tensors (B,3,H,W) in [-1,1].
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualCNN(nn.Module):
    """Detail-restoration network: conv head -> residual blocks -> refinement -> recon."""

    def __init__(self, in_channels: int = 3, feature_channels: int = 32,
                 num_blocks: int = 3, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size, padding=pad),
            nn.PReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(feature_channels, kernel_size) for _ in range(num_blocks)]
        )
        self.refine = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size, padding=pad),
            nn.BatchNorm2d(feature_channels),
            nn.PReLU(),
        )
        self.reconstruct = nn.Conv2d(feature_channels, in_channels, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input (B,3,H,W) in [-1,1]; output restored image of same shape (global residual)."""
        feat = self.head(x)
        feat = self.res_blocks(feat)
        feat = self.refine(feat)
        out = self.reconstruct(feat)
        return torch.tanh(out + x)          # global residual connection


if __name__ == "__main__":
    m = ResidualCNN()
    x = torch.randn(1, 3, 320, 320)
    print(m(x).shape)
