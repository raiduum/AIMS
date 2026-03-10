import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def preprocess_point_cloud(
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


def open3d_to_trimesh(mesh_o3d: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """
    Convert Open3D mesh to trimesh.
    """
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Mesh has no vertices or faces.")

    mesh_tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return mesh_tm


def compute_volume(mesh_tm: trimesh.Trimesh) -> float:
    """
    Compute volume from a trimesh mesh.
    """
    if not mesh_tm.is_watertight:
        raise ValueError(
            "Mesh is not watertight. Volume may be invalid. "
            "Try tuning Poisson depth / density trimming / preprocessing."
        )
    return float(mesh_tm.volume)


def main():

    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Point cloud -> Poisson mesh -> volume")
    parser.add_argument("--i", required=True, help="Input point cloud path")
    parser.add_argument("--o", default="poisson_mesh.ply", help="Output mesh path")
    parser.add_argument("--voxel_size", type=float, default=0.00, help="Voxel downsample size")
    parser.add_argument("--depth", type=int, default=6, help="Poisson depth")
    parser.add_argument("--density_quantile", type=float, default=0.00, help="Trim lowest-density vertices by quantile (0~1)",)
    args = parser.parse_args()

    input_path = Path(args.i+".ply")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    load = time.perf_counter()
    print(f"[1/5] Loading point cloud: {input_path}", "running_time_load: ", load - start)
    pcd = o3d.io.read_point_cloud(input_path)
    if pcd.is_empty():
        raise ValueError(f"Failed to load point cloud or file is empty: {input_path}")
    print(f"  points: {len(np.asarray(pcd.points))}")

    preprocessing = time.perf_counter()
    print("[2/5] Preprocessing point cloud")
    pcd = preprocess_point_cloud(
        pcd,
        voxel_size=args.voxel_size if args.voxel_size > 0 else None,
    )
    print(f"  points after preprocessing: {len(np.asarray(pcd.points))}", "running_time_preprocessing: ", preprocessing - load)


    estimate = time.perf_counter()
    print("[3/5] Estimating normals")
    pcd = ensure_normals(pcd)
    print("estimate_running_time: ", estimate - preprocessing)
    
    print("[4/5] Poisson reconstruction")
    mesh_o3d = poisson_reconstruct(
        pcd,
        depth=args.depth,
        density_quantile=args.density_quantile,
    )
    print(
        f"  mesh vertices: {len(np.asarray(mesh_o3d.vertices))}, "
        f"faces: {len(np.asarray(mesh_o3d.triangles))}"
    )

    o3d.io.write_triangle_mesh(args.o, mesh_o3d)
    making_mesh = time.perf_counter()
    print(f"  saved mesh to: {args.o}" "making_mesh_running_time: ", making_mesh - estimate)


    measure_volume = time.perf_counter()
    print("[5/5] Computing volume with trimesh")
    mesh_tm = open3d_to_trimesh(mesh_o3d)
    print(f"  watertight: {mesh_tm.is_watertight}")
    volume = compute_volume(mesh_tm)

    print("\n=== RESULT ===")
    print(f"Volume: {abs(volume):.6f}", "measure_volume_running_time: ", measure_volume - making_mesh, "total_running_time: ", measure_volume - start)
    with open(args.i+".txt", "r") as f:
        real_volume = float(f.read().strip())
    print("diffrence %:", (abs(abs(volume) - real_volume)/real_volume)*100)


if __name__ == "__main__":
    main()