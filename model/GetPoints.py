import torch
import torch.nn as nn


class GetPoints(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU()
        )

        self.fc = nn.Linear(64*64*64, 3000*3)

    def forward(self, x):

        feat = self.encoder(x)
        feat = feat.view(feat.size(0), -1)

        points = self.fc(feat)
        points = points.view(-1,3000,3)

        return points