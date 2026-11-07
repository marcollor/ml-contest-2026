import torch
from torch import nn


class ParallelConv4(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Conv1d(36, 48, 3, stride=2, padding=1, bias=False)
        self.context = nn.Conv1d(36, 24, 3, stride=2, padding=4, dilation=4, bias=False)

    def forward(self, x):
        return torch.cat((self.local(x), self.context(x)), dim=1)


class TemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        ci = 4
        for co, kernel in [(24, 9), (24, 7), (36, 5)]:
            layers.extend(
                [
                    nn.Conv1d(
                        ci, co, kernel, stride=2, padding=kernel // 2, bias=False
                    ),
                    nn.BatchNorm1d(co),
                    nn.ReLU(),
                ]
            )
            ci = co
        layers.extend([ParallelConv4(), nn.BatchNorm1d(72), nn.ReLU()])
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(144, 8)

    def forward(self, x):
        z = self.body(x)
        mean = z.mean(dim=-1)
        std = torch.sqrt(z.var(dim=-1, unbiased=False) + 1e-6)
        return self.fc(torch.cat((mean, std), dim=1))


if __name__ == "__main__":
    model = TemporalCNN().eval()
    assert sum(p.numel() for p in model.parameters()) == 18464
    assert model(torch.zeros(1, 4, 256)).shape == (1, 8)
    print("18464 parameters; input (1,4,256), output (1,8).")
