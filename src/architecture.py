"""Compact residual temporal CNN with filtered downsampling and dilated blocks."""

import torch
from torch import nn
from torch.nn import functional as F


class LowPassDownsample(nn.Module):
    """Apply a fixed [1, 2, 1] / 4 filter per channel before decimation."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        kernel = torch.tensor([1.0, 2.0, 1.0]).view(1, 1, 3) / 4.0
        self.register_buffer("kernel", kernel.repeat(channels, 1, 1))

    def forward(self, x):
        # Replication preserves constant signals at the window boundaries.
        x = F.pad(x, (1, 1), mode="replicate")
        return F.conv1d(x, self.kernel, stride=2, groups=self.channels)


class ResidualTemporalBlock(nn.Module):
    """Depthwise temporal filtering followed by pointwise channel mixing."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size=5, padding=2 * dilation,
                dilation=dilation, groups=channels, bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.body(x))


class BaseTemporalCNN(nn.Module):
    """Accept (N, 4, 256) normalized signals and return (N, 8) logits."""

    def __init__(self):
        super().__init__()
        channels = 32
        self.body = nn.Sequential(
            nn.Conv1d(4, channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            LowPassDownsample(channels),
            ResidualTemporalBlock(channels, dilation=1),
            LowPassDownsample(channels),
            ResidualTemporalBlock(channels, dilation=2),
            LowPassDownsample(channels),
            ResidualTemporalBlock(channels, dilation=4),
            ResidualTemporalBlock(channels, dilation=8),
        )
        self.fc = nn.Linear(2 * channels, 8)

    def forward(self, x):
        z = self.body(x)
        mean = z.mean(dim=-1)
        std = torch.sqrt(z.var(dim=-1, unbiased=False) + 1e-6)
        return self.fc(torch.cat((mean, std), dim=1))


class TemporalCNN(BaseTemporalCNN):
    """Combine pooled temporal features and lagged signal relations."""

    def __init__(self):
        super().__init__()
        self.relations = nn.Sequential(nn.Linear(30, 16), nn.ReLU())
        self.fc = nn.Linear(80, 8)

    def forward(self, x):
        z = self.body(x)
        pooled = torch.cat((z.mean(-1), torch.sqrt(z.var(-1, unbiased=False)+1e-6)), 1)
        values = []
        for lag in (0, 2, 8):
            for a in range(4):
                for b in range(a+1, 4):
                    left = x[:, a] if lag == 0 else x[:, a, :-lag]
                    right = x[:, b] if lag == 0 else x[:, b, lag:]
                    values.append((left*right).mean(-1))
        for lag in (1, 4, 16):
            for channel in range(4):
                values.append((x[:,channel,:-lag]*x[:,channel,lag:]).mean(-1))
        relations = self.relations(torch.stack(values, 1))
        return self.fc(torch.cat((pooled, relations), 1))
