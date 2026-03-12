import torch
import torch.nn as nn


class Retouch(nn.Module):

    def __init__(self, max_points=6000):
        super().__init__()
        self.max_points = max_points

    def restoration(self, points):
        # TODO: occlusion / truncation-aware restoration
        return points

    def scale_expand(self, points, scale_factor=1.2):
        center = points.mean(dim=1, keepdim=True)
        expanded = center + (points - center) * scale_factor
        return expanded

    def density_completion(self, points):
        b, n, _ = points.shape

        if n >= self.max_points:
            return points

        add = self.max_points - n

        idx = torch.randint(0, n, (b, add), device=points.device)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, 3)

        sampled = torch.gather(points, 1, idx_expanded)
        noise = torch.randn_like(sampled) * 0.005
        new_points = sampled + noise

        return torch.cat([points, new_points], dim=1)

    def forward(self, points, flags):
        """
        points: [B, N, 3] or [1, N, 3]
        flags: [restoration, scale_expand, density_completion]
        """

        restoration = bool(flags[0].item() if torch.is_tensor(flags[0]) else flags[0])
        scale_expand = bool(flags[1].item() if torch.is_tensor(flags[1]) else flags[1])
        density = bool(flags[2].item() if torch.is_tensor(flags[2]) else flags[2])

        if restoration:
            return self.restoration(points)

        elif scale_expand:
            return self.scale_expand(points)

        elif density:
            return self.density_completion(points)

        return points