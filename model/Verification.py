import torch
import torch.nn as nn


class Verification(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(3000*3,256),
            nn.ReLU(),
            nn.Linear(256,3)
        )

    def forward(self, points):

        x = points.view(points.size(0),-1)
        out = self.fc(x)

        return torch.sigmoid(out)