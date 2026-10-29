"""Define the BornNet class, which builds BornNet, LinearBornNet, and BornAE."""

from __future__ import annotations

import torch
from torch import nn

from blocks import Downsample, ResBlock, Upsample


class BornNet(nn.Module):
    """``kind`` chooses BornNet, LinearBornNet, or BornAE.

    ``inverse`` maps the 20,000 stacked real and imaginary measurements to the
    10,000 contrast pixels. ``forward`` maps the pixels to the same stacked
    measurements. Neither matrix includes ``data_scale``. Divide the inputs by
    ``data_scale`` before passing them to the model.
    """

    def __init__(
        self,
        kind: str,
        inverse: torch.Tensor | None = None,
        forward: torch.Tensor | None = None,
        data_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if kind not in ("BornNet", "LinearBornNet", "BornAE"):
            raise ValueError(f"Unknown model: {kind}")
        if kind == "BornNet" and inverse is None:
            raise ValueError("BornNet requires an inverse operator")
        if kind == "BornAE" and forward is None:
            raise ValueError("BornAE requires a forward operator")
        self.kind = kind
        self.data_scale = data_scale
        self.register_buffer(
            "inverse", inverse.detach().float() if inverse is not None else None,
            persistent=False,
        )
        self.register_buffer(
            "forward_operator", forward.detach().float() if forward is not None else None,
            persistent=False,
        )

        self.input_proj = nn.Conv2d(2, 32, 3, padding=1)
        widths = (32, 64, 64, 64)
        strides = (2, 2, 5)
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        for level, width in enumerate(widths):
            attention = level in (1, 2)
            down = nn.ModuleList([
                ResBlock(width, width, attention=attention),
                ResBlock(width, width, attention=attention),
            ])
            if level < 3:
                down.append(Downsample(width, widths[level + 1], strides[level]))
            self.down.append(down)

            up = nn.ModuleList([
                ResBlock(2 * width, width, attention=attention),
                ResBlock(2 * width, width, attention=attention),
            ])
            if level > 0:
                up.append(Upsample(width, widths[level - 1], strides[level - 1]))
            self.up.insert(0, up)

        self.mid = nn.Sequential(
            ResBlock(64, 64, attention=True),
            ResBlock(64, 64),
        )
        self.born_input = nn.Sequential(
            nn.GroupNorm(32, 64), nn.SiLU(), nn.Conv2d(64, 2, 3, padding=1)
        )
        if kind != "BornNet":
            self.born_output = nn.Sequential(
                nn.Linear(20000, 1024, bias=False),
                nn.Linear(1024, 10000, bias=False),
            )
        self.resnet = nn.Sequential(
            ResBlock(1, 32, groups=1), ResBlock(32, 32), ResBlock(32, 32)
        )
        self.final = nn.Sequential(
            nn.GroupNorm(32, 32), nn.SiLU(), nn.Conv2d(32, 1, 3, padding=1)
        )
        if kind == "BornAE":
            self.corrector = nn.Sequential(
                ResBlock(2, 32, groups=1),
                ResBlock(32, 32),
                nn.GroupNorm(32, 32),
                nn.SiLU(),
                nn.Conv2d(32, 2, 3, padding=1),
            )

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.input_proj(values)
        features = stem
        saved = []
        for level, blocks in enumerate(self.down):
            for block in blocks[:2]:
                features = block(features)
                saved.append(features)
            if level < 3:
                features = blocks[2](features)
        features = self.mid(features)
        for level, blocks in enumerate(self.up):
            for block in blocks[:2]:
                features = block(torch.cat((features, saved.pop()), dim=1))
            if level < 3:
                features = blocks[2](features)
        corrected = self.born_input(torch.cat((features, stem), dim=1)).flatten(1)
        if self.kind == "BornNet":
            with torch.autocast(device_type=values.device.type, enabled=False):
                intermediate = (corrected.float() @ self.inverse.T) * self.data_scale
        else:
            intermediate = self.born_output(corrected)
        intermediate = intermediate.reshape(-1, 1, 100, 100)
        image = self.final(self.resnet(intermediate))
        return image, intermediate

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image, intermediate = self.encode(values)
        if self.kind != "BornAE":
            return image, intermediate
        with torch.autocast(device_type=values.device.type, enabled=False):
            field = (image.float().flatten(1) @ self.forward_operator.T) / self.data_scale
        return image, self.corrector(field.reshape(-1, 2, 100, 100))
