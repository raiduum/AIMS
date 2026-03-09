import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import optuna


def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float | None = None,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    work = pcd
    if voxel_size is not None and voxel_size > 0:
        work = work.voxel_down_sample(voxel_size=voxel_size)

    work, _ = work.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return work


def ensure_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float | None = None,
    max_nn: int = 30,
) -> o3d.geometry.PointCloud:
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
    pcd.orient_normals_consistent_tangent_plane(k=min(50, len(pts) - 1))
    return pcd


def poisson_reconstruct(
    pcd: o3d.geometry.PointCloud,
    depth: int = 6,
    density_quantile: float = 0.0,
) -> o3d.geometry.TriangleMesh:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
    )

    densities = np.asarray(densities)
    if len(densities) == 0:
        raise ValueError("Poisson reconstruction returned empty density array.")

    if density_quantile > 0:
        threshold = np.quantile(densities, density_quantile)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.compute_vertex_normals()
    return mesh


def open3d_to_trimesh(mesh_o3d: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Mesh has no vertices or faces.")

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def evaluate_once(ply_path: Path, real_volume: float, voxel_size: float, depth: int, density_quantile: float):
    t0 = time.perf_counter()

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError("Empty point cloud")

    pcd = preprocess_point_cloud(
        pcd,
        voxel_size=voxel_size if voxel_size > 0 else None,
    )
    pcd = ensure_normals(pcd)

    mesh_o3d = poisson_reconstruct(
        pcd,
        depth=depth,
        density_quantile=density_quantile,
    )

    mesh_tm = open3d_to_trimesh(mesh_o3d)

    if not mesh_tm.is_watertight or not mesh_tm.is_volume:
        raise ValueError("Invalid mesh for volume")

    pred_volume = abs(float(mesh_tm.volume))
    rel_err = abs(pred_volume - real_volume) / real_volume
    runtime = time.perf_counter() - t0

    return rel_err, pred_volume, runtime


def make_objective(ply_path: Path, real_volume: float, runtime_weight: float):
    def objective(trial: optuna.trial.Trial) -> float:
        voxel_size = trial.suggest_categorical("voxel_size", [0.0, 0.002, 0.005, 0.01])
        depth = trial.suggest_int("depth", 5, 8)
        density_quantile = trial.suggest_categorical(
            "density_quantile", [0.0, 0.005, 0.01, 0.02]
        )

        try:
            rel_err, pred_volume, runtime = evaluate_once(
                ply_path, real_volume, voxel_size, depth, density_quantile
            )
            score = rel_err + runtime_weight * runtime

            trial.set_user_attr("pred_volume", pred_volume)
            trial.set_user_attr("rel_err", rel_err)
            trial.set_user_attr("runtime", runtime)
            return score

        except Exception as e:
            trial.set_user_attr("error", str(e))
            return 1e6  # 실패 trial에 큰 패널티

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", required=True, help="sample prefix without extension")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--runtime_weight", type=float, default=0.01)
    parser.add_argument("--study_name", type=str, default="poisson_opt")
    parser.add_argument("--storage", type=str, default="sqlite:///poisson_opt.db")
    args = parser.parse_args()

    ply_path = Path(args.i + ".ply")
    txt_path = Path(args.i + ".txt")

    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    if not txt_path.exists():
        raise FileNotFoundError(txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        real_volume = float(f.read().strip())

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
    )

    study.optimize(
        make_objective(ply_path, real_volume, args.runtime_weight),
        n_trials=args.n_trials,
    )

    best = study.best_trial
    print("\n=== BEST RESULT ===")
    print("best score:", best.value)
    print("best params:", best.params)
    print("best rel_err:", best.user_attrs.get("rel_err"))
    print("best pred_volume:", best.user_attrs.get("pred_volume"))
    print("best runtime:", best.user_attrs.get("runtime"))


if __name__ == "__main__":
    main()