import torch
import torch.nn as nn


class Retouch(nn.Module):

    def __init__(self, max_points=6000):
        super().__init__()

        self.max_points = max_points

    def restoration(self, points):

        # placeholder restoration
        return points

    def scale_expand(self, points):

        return points * 1.2

    def density_completion(self, points):

        b,n,_ = points.shape

        if n >= self.max_points:
            return points

        add = self.max_points - n
        noise = torch.randn(b,add,3).to(points.device)*0.01

        return torch.cat([points,noise],dim=1)

    def forward(self, points, flags):

        restoration, scale_expand, density = flags

        if restoration:
            return self.restoration(points)

        elif scale_expand:
            return self.scale_expand(points)

        elif density:
            return self.density_completion(points)

        return points