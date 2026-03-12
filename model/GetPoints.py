import torch
import torch.nn as nn


class GetPoints(nn.Module):

    def __init__(self, num_points: int = 3000):
        super().__init__()
        self.num_points = num_points

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),    # 512 -> 256
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 256 -> 128
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=2, padding=1), # 64 -> 32
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_points * 3)
        )

    def forward(self, x):
        feat = self.encoder(x)
        points = self.fc(feat)
        points = points.reshape(x.size(0), self.num_points, 3)
        return points