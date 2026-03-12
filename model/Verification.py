import torch
import torch.nn as nn


class Verification(nn.Module):

    def __init__(self):
        super().__init__()

        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

    def forward(self, points):
        # points: [B, 3000, 3]

        feat = self.point_mlp(points)          # [B, 3000, 128]
        feat = feat.max(dim=1).values          # [B, 128]  global max pooling

        out = self.head(feat)                  # [B, 3]
        return torch.sigmoid(out)