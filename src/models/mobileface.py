from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List

import torch
from torch import Tensor, nn


def _make_divisible(value: float, divisor: int = 8) -> int:
    """Round channel dimensions to be divisible by a given divisor."""
    min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class ConvBNAct(nn.Sequential):
    """Convolution block with BatchNorm and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class InvertedResidual(nn.Module):
    """Inverted residual block with depthwise separable convolutions."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expansion: int) -> None:
        super().__init__()
        hidden_dim = _make_divisible(in_channels * expansion)
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []
        if expansion != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1))
        layers.append(ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.block(x)
        if self.use_residual:
            out = out + identity
        return out


@dataclass
class MobileFaceNetConfig:
    """Configuration describing MobileFaceNet channel layout."""

    input_channels: int
    layers: Sequence[int]
    expansion: Sequence[int]
    strides: Sequence[int]

    def validate(self) -> None:
        lengths = {len(self.layers), len(self.expansion), len(self.strides)}
        if len(lengths) != 1:
            raise ValueError("Layer, expansion, and stride definitions must share the same length.")


class MobileFaceNet(nn.Module):
    """Lightweight face embedding network.

    Parameters
    ----------
    embedding_dim:
        Dimension of the output embedding vector.
    width_multiplier:
        Scales the base number of channels to trade accuracy for latency.
    dropout:
        Dropout probability applied before the embedding projection.
    config:
        Configuration object specifying backbone stages. Defaults to the canonical layout.

    Returns
    -------
    torch.Tensor
        L2-normalised embedding tensor of shape `(batch, embedding_dim)`.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        width_multiplier: float = 1.0,
        dropout: float = 0.1,
        config: MobileFaceNetConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = MobileFaceNetConfig(
                input_channels=32,
                layers=[64, 64, 128, 128, 256, 256, 512],
                expansion=[1, 2, 2, 4, 4, 4, 4],
                strides=[2, 1, 2, 1, 2, 1, 1],
            )
        config.validate()

        stem_channels = _make_divisible(16 * width_multiplier)
        self.stem = nn.Sequential(
            ConvBNAct(3, stem_channels, kernel_size=3, stride=2),
            ConvBNAct(stem_channels, config.input_channels, kernel_size=3, stride=1, groups=stem_channels),
        )

        backbone: List[nn.Module] = []
        in_channels = config.input_channels
        for layer_channels, expansion, stride in zip(config.layers, config.expansion, config.strides):
            out_channels = _make_divisible(layer_channels * width_multiplier)
            backbone.append(InvertedResidual(in_channels, out_channels, stride=stride, expansion=max(1, expansion)))
            in_channels = out_channels
        self.backbone = nn.Sequential(*backbone)

        embedding_input = _make_divisible(in_channels)
        self.head = nn.Sequential(
            ConvBNAct(embedding_input, embedding_input * 2, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.embedding = nn.Linear(embedding_input * 2, embedding_dim, bias=True)
        self.output_norm = nn.functional.normalize

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        features = self.stem(x)
        features = self.backbone(features)
        features = self.head(features)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        embeddings = self.embedding(features)
        return self.output_norm(embeddings, dim=1)


def build_mobileface(embedding_dim: int = 128, width_multiplier: float = 1.0, dropout: float = 0.1) -> MobileFaceNet:
    """Factory helper for the default MobileFaceNet variant."""
    return MobileFaceNet(embedding_dim=embedding_dim, width_multiplier=width_multiplier, dropout=dropout)
