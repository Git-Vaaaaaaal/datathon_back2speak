"""PyTorch model definitions for mel-spectrogram CNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d → BN → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int] = (2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectrogramCNN(nn.Module):
    """
    Small CNN that takes a log-mel spectrogram (1, n_mels, time) and
    outputs a binary probability.

    Architecture
    ------------
    Conv(1→32) → Conv(32→64) → Conv(64→128) → AdaptiveAvgPool → FC(256→1)
    """

    def __init__(self, n_mels: int = 128, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,  32, pool=(2, 2)),
            ConvBlock(32, 64, pool=(2, 2)),
            ConvBlock(64, 128, pool=(2, 2)),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 1, n_mels, time_frames)

        Returns
        -------
        logits : (batch,)  — apply sigmoid for probability
        """
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


class ResidualBlock(nn.Module):
    """Simple residual block for a slightly deeper CNN."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class SpectrogramResCNN(nn.Module):
    """
    CNN with residual connections — better gradient flow for small datasets.
    """

    def __init__(self, n_mels: int = 128, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer1 = nn.Sequential(ResidualBlock(32), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResidualBlock(64), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)
