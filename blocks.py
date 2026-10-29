"""Define the convolutional and attention blocks that the three networks share."""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class Downsample(nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            inputs, outputs, stride + 1, stride=stride, padding=stride // 2
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(values)


class Upsample(nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(inputs, outputs, stride, stride=stride)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.upsample(values)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.head_dim = channels // 4
        self.norm = nn.GroupNorm(32, channels)
        self.Q = nn.Conv2d(channels, channels, 1)
        self.K = nn.Conv2d(channels, channels, 1)
        self.V = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.norm(values)
        query, key, value = (
            rearrange(
                projection(features),
                "b (head_dim heads) h w -> b heads (h w) head_dim",
                head_dim=self.head_dim,
                heads=4,
            )
            for projection in (self.Q, self.K, self.V)
        )
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = rearrange(
            attended,
            "b heads (h w) head_dim -> b (head_dim heads) h w",
            heads=4,
            head_dim=self.head_dim,
            h=values.shape[2],
            w=values.shape[3],
        )
        return (values + self.proj(attended)) / math.sqrt(2)


def make_block(inputs: int, outputs: int, groups: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.GroupNorm(groups, inputs),
        nn.SiLU(),
        nn.Dropout(dropout) if dropout else nn.Identity(),
        nn.Conv2d(inputs, outputs, 3, padding=1),
    )


class ResBlock(nn.Module):
    def __init__(
        self,
        inputs: int,
        outputs: int,
        groups: int = 32,
        attention: bool = False,
    ) -> None:
        super().__init__()
        self.skip_connection = (
            nn.Identity() if inputs == outputs else nn.Conv2d(inputs, outputs, 1)
        )
        self.block1 = make_block(inputs, outputs, groups, dropout=0)
        self.block2 = make_block(outputs, outputs, groups, dropout=0.1)
        self.attn = AttentionBlock(outputs) if attention else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.block2(self.block1(values))
        features = (self.skip_connection(values) + features) / math.sqrt(2)
        return self.attn(features)
