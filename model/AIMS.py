import time

import numpy as np
import open3d as o3d
import trimesh

import torch.nn as nn
import torch
import torch.nn.functional as F

from model.GetPoints import GetPoints
from model.Verification import Verification
from model.Retouch import Retouch


class AIMS(nn.Module):

    def __init__(self):
        super().__init__()

        self.get_points = GetPoints()
        self.verification = Verification()
        self.retouch = Retouch()

    def forward(self, image):
        raw_points = self.get_points(image)                 # [B, N, 3]
        verify_scores = self.verification(raw_points)       # [B, 3] or [B, K]

        final_points_list = []
        retouch_type_list = []

        for i in range(raw_points.shape[0]):
            scores = verify_scores[i]

            # 예시:
            # scores[0] = restoration
            # scores[1] = small_object_expand
            # scores[2] = density_completion
            # 하나만 선택
            mode = torch.argmax(scores).item() + 1
            max_score = torch.max(scores).item()

            if max_score < 0.5:
                mode = 0

            if mode == 0:
                final_points = raw_points[i:i+1]
            else:
                final_points = self.retouch(raw_points[i:i+1], mode=mode)

            final_points_list.append(final_points)
            retouch_type_list.append(mode)

        final_points = torch.cat(final_points_list, dim=0)
        retouch_type = torch.tensor(retouch_type_list, device=raw_points.device)

        return {
            "raw_points": raw_points,
            "final_points": final_points,
            "verify_scores": verify_scores,
            "retouch_type": retouch_type,
        }

    def relative_volume_error(self, pred_volume: torch.Tensor, gt_volume: torch.Tensor, eps: float = 1e-8):
        gt_volume = torch.clamp(gt_volume, min=eps)
        return ((pred_volume - gt_volume) / gt_volume)*100


    def calcultate_volume(self, points: torch.Tensor, eps: float = 1e-8):
        """
        points: [B, N, 3]
        return: [B]
        """
        volumes = []

        for i in range(points.shape[0]):
            p = points[i].detach().cpu().numpy()

            vol = self.cal_volume(p)

            volumes.append(float(vol))

        return torch.tensor(volumes, dtype=torch.float32)
    
    def preprocess_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float | None = None,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
    ) -> o3d.geometry.PointCloud:
        """
        Optional downsampling + statistical outlier removal.
        """
        work = pcd

        if voxel_size is not None and voxel_size > 0:
            work = work.voxel_down_sample(voxel_size=voxel_size)

        work, _ = work.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        return work


    def ensure_normals(
        self,
        pcd: o3d.geometry.PointCloud,
        radius: float | None = None,
        max_nn: int = 30,
    ) -> o3d.geometry.PointCloud:
        """
        Estimate and orient normals if missing.
        Poisson reconstruction needs oriented normals.
        """
        pts = np.asarray(pcd.points)
        if len(pts) < 10:
            raise ValueError("Too few points for normal estimation.")

        if radius is None:
            bounds = pts.max(axis=0) - pts.min(axis=0)
            diag = float(np.linalg.norm(bounds))
            radius = max(diag * 0.02, 1e-6)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius,
                max_nn=max_nn,
            )
        )

        # Try to orient normals consistently for Poisson
        pcd.orient_normals_consistent_tangent_plane(k=min(50, len(pts) - 1))
        return pcd


    def poisson_reconstruct(
        self,
        pcd: o3d.geometry.PointCloud,
        depth: int = 9,
        density_quantile: float = 0.02,
    ) -> o3d.geometry.TriangleMesh:
        """
        Reconstruct a watertight-ish mesh using Poisson reconstruction.
        Then trim low-density vertices to reduce extrapolated artifacts.
        """
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
        )

        densities = np.asarray(densities)
        if len(densities) == 0:
            raise ValueError("Poisson reconstruction returned empty density array.")

        # Remove lowest-density vertices that are often extrapolated outside the object
        threshold = np.quantile(densities, density_quantile)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        mesh.compute_vertex_normals()
        return mesh


    def open3d_to_trimesh(self, mesh_o3d: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
        """
        Convert Open3D mesh to trimesh.
        """
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)

        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError("Mesh has no vertices or faces.")

        mesh_tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        return mesh_tm


    def compute_volume(self, mesh_tm: trimesh.Trimesh) -> float:
        """
        Compute volume from a trimesh mesh.
        """
        if not mesh_tm.is_watertight:
            raise ValueError(
                "Mesh is not watertight. Volume may be invalid. "
                "Try tuning Poisson depth / density trimming / preprocessing."
            )
        return float(mesh_tm.volume)
    
    def cal_volume(self, points):

        pcd = points
        pcd = self.preprocess_point_cloud(
            pcd,
            voxel_size=0.01,
        )

        mesh_o3d = self.poisson_reconstruct(
            pcd,
            depth=7,
            density_quantile=0.0,
        )

        mesh_tm = self.open3d_to_trimesh(mesh_o3d)
        volume = self.compute_volume(mesh_tm)

 
        return volume

    def point_count_loss(self, points: torch.Tensor, min_points: int = 3000, max_points: int = 6000):
        n = points.shape[1]
        if min_points <= n <= max_points:
            return points.new_tensor(0.0)
        if n < min_points:
            return points.new_tensor((min_points - n) / float(min_points))
        return points.new_tensor((n - max_points) / float(max_points))


    def density_uniformity_loss(self, points: torch.Tensor):
        """
        초기형: 중심 기준 반경 분포의 분산으로 균일성 거칠게 측정
        이후 kNN 기반으로 바꾸면 더 좋음
        """
        center = points.mean(dim=1, keepdim=True)                  # [B,1,3]
        dist = torch.norm(points - center, dim=-1)                 # [B,N]
        mean_dist = dist.mean(dim=1, keepdim=True)
        var = ((dist - mean_dist) ** 2).mean(dim=1)                # [B]
        return var.mean()


    def outlier_noise_loss(self, points: torch.Tensor):
        center = points.mean(dim=1, keepdim=True)
        dist = torch.norm(points - center, dim=-1)
        mean = dist.mean(dim=1, keepdim=True)
        std = dist.std(dim=1, keepdim=True)
        threshold = mean + 2.0 * std
        penalty = torch.relu(dist - threshold)
        return penalty.mean()


    def retouch_consistency_loss(self, raw_points: torch.Tensor, final_points: torch.Tensor):
        """
        조건부 모델이 과도하게 형상을 망가뜨리지 않게 억제
        point 개수가 다를 수 있으므로 bbox 기반으로 비교
        """
        raw_min = raw_points.min(dim=1).values
        raw_max = raw_points.max(dim=1).values
        fin_min = final_points.min(dim=1).values
        fin_max = final_points.max(dim=1).values

        return F.l1_loss(fin_min, raw_min) + F.l1_loss(fin_max, raw_max)

    
    def loss_fn(
        self, 
        outputs: dict,
        gt_volume: torch.Tensor,
        lambda_proxy_vol: float = 1.0,
        lambda_count: float = 0.05,
        lambda_density: float = 0.03,
        lambda_noise: float = 0.02,
        lambda_retouch: float = 0.02,
    ):
        """
        outputs keys:
        - raw_points:   [B, N, 3]
        - final_points: [B, M, 3]
        - verify_scores: optional
        - retouch_type: optional

        gt_volume: [B]
        real_volume_fn:
        Python callable. final_points -> pred_volume_real [B]
        비미분 가능해도 됨. 대신 detach guidance로만 사용.
        """
        raw_points = outputs["raw_points"]
        final_points = outputs["final_points"]

        gt_volume = gt_volume.float().view(-1).to(final_points.device)

        # 1) differentiable proxy volume
        pred_volume_proxy = self.calcultate_volume(final_points)
        proxy_vol_err = self.relative_volume_error(pred_volume_proxy, gt_volume).mean()

        # 2) structural losses
        count_l = self.point_count_loss(final_points)
        density_l = self.density_uniformity_loss(final_points)
        noise_l = self.outlier_noise_loss(final_points)
        retouch_l = self.retouch_consistency_loss(raw_points, final_points)

        # 3) actual poisson volume (optional, non-diff allowed)
        real_vol_err = final_points.new_tensor(0.0)
        pred_volume_real_mean = final_points.new_tensor(0.0)

        total_loss = (
                lambda_proxy_vol * proxy_vol_err
                + lambda_count * count_l
                + lambda_density * density_l
                + lambda_noise * noise_l
                + lambda_retouch * retouch_l
        )

        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_proxy_vol": proxy_vol_err.detach(),
            "loss_count": count_l.detach(),
            "loss_density": density_l.detach(),
            "loss_noise": noise_l.detach(),
            "loss_retouch": retouch_l.detach(),
            "loss_real_vol": real_vol_err.detach(),
            "pred_proxy_volume_mean": pred_volume_proxy.mean().detach(),
            "pred_real_volume_mean": pred_volume_real_mean.detach(),
        }

        return total_loss, loss_dict
    def train_step(
            self,
            step,
            image,
            gt_volume,
            model,
            optimizer,
            scheduler,
            use_amp: bool = True,
            ):
        model.train()
        with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float32):
            loss, loss_dict = self.loss_fn(
                outputs=self.forward(image),
                gt_volume=gt_volume,
                lambda_proxy_vol=1.0,
                lambda_count=0.05,
                lambda_density=0.03,
                lambda_noise=0.02,
                lambda_retouch=0.02,
                )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return loss_dict