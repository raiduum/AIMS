import random
from pathlib import Path
import open3d as o3d
import numpy as np
import trimesh


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def random_rotation_matrix():
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0, 2 * np.pi)

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ])


def apply_random_pose(mesh):
    mesh = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = random_rotation_matrix()
    mesh.apply_transform(T)
    mesh.vertices -= mesh.centroid
    return mesh


def make_box():
    extents = [
        np.random.uniform(3.5, 12.0),
        np.random.uniform(2.5, 9.0),
        np.random.uniform(2.0, 8.0),
    ]
    return trimesh.creation.box(extents=extents)


def make_long_box():
    extents = [
        np.random.uniform(9.0, 18.0),
        np.random.uniform(1.8, 4.0),
        np.random.uniform(1.8, 4.0),
    ]
    return trimesh.creation.box(extents=extents)


def make_flat_box():
    extents = [
        np.random.uniform(5.0, 13.0),
        np.random.uniform(4.0, 11.0),
        np.random.uniform(0.8, 2.2),
    ]
    return trimesh.creation.box(extents=extents)


def make_sphere():
    radius = np.random.uniform(1.8, 5.0)
    return trimesh.creation.icosphere(subdivisions=4, radius=radius)


def make_ellipsoid():
    mesh = trimesh.creation.icosphere(
        subdivisions=4,
        radius=np.random.uniform(1.2, 3.0)
    )
    scales = np.array([
        np.random.uniform(1, 2.0),
        np.random.uniform(1, 1.8),
        np.random.uniform(1, 2.0),
    ])
    mesh.apply_scale(scales)
    return mesh


def make_cylinder():
    radius = np.random.uniform(1.2, 3.5)
    height = np.random.uniform(3.0, 12.0)
    return trimesh.creation.cylinder(radius=radius, height=height, sections=64)


def make_tall_cylinder():
    radius = np.random.uniform(1.0, 2.2)
    height = np.random.uniform(9.0, 18.0)
    return trimesh.creation.cylinder(radius=radius, height=height, sections=64)


def make_squat_cylinder():
    radius = np.random.uniform(2.0, 4.5)
    height = np.random.uniform(1.5, 3.5)
    return trimesh.creation.cylinder(radius=radius, height=height, sections=64)


def make_capsule():
    radius = np.random.uniform(1.0, 2.5)
    height = np.random.uniform(3.5, 12.0)
    return trimesh.creation.capsule(radius=radius, height=height, count=[32, 32])


def make_cone():
    radius = np.random.uniform(1.5, 4.5)
    height = np.random.uniform(2.5, 10.0)
    return trimesh.creation.cone(radius=radius, height=height, sections=64)


def make_torus():
    major_radius = np.random.uniform(2.5, 6.0)
    minor_radius = np.random.uniform(0.6, 1.8)
    return trimesh.creation.torus(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_sections=64,
        minor_sections=32,
    )


def make_rounded_box():
    # box의 convex hull with sphere 느낌으로 약간 둥근 형태 대체
    box = trimesh.creation.box(extents=[
        np.random.uniform(3.5, 10.0),
        np.random.uniform(2.5, 8.0),
        np.random.uniform(2.0, 7.0),
    ])
    sphere = trimesh.creation.icosphere(
        subdivisions=3,
        radius=np.random.uniform(1.0, 2.0)
    )
    sphere.vertices += np.array([
        np.random.uniform(-0.5, 0.5),
        np.random.uniform(-0.5, 0.5),
        np.random.uniform(-0.5, 0.5),
    ])
    return trimesh.util.concatenate([box, sphere]).convex_hull


SHAPES = {
    "box": make_box,
    "long_box": make_long_box,
    "flat_box": make_flat_box,
    "sphere": make_sphere,
    "ellipsoid": make_ellipsoid,
    "cylinder": make_cylinder,
    "tall_cylinder": make_tall_cylinder,
    "squat_cylinder": make_squat_cylinder,
    "capsule": make_capsule,
    "cone": make_cone,
    "torus": make_torus,
    "rounded_box": make_rounded_box,
}


def sample_point_cloud(mesh, n_points=100_000, noise_std=0.0):
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    if noise_std > 0:
        points += np.random.normal(0, noise_std, size=points.shape)
    return points


def generate_one(shape_name=None, n_points=100_000, noise_std=0.0):
    if shape_name is None:
        shape_name = random.choice(list(SHAPES.keys()))

    mesh = SHAPES[shape_name]()
    mesh = apply_random_pose(mesh)

    points = sample_point_cloud(mesh, n_points=n_points, noise_std=noise_std)
    volume = float(mesh.volume)

    return shape_name, points, volume


def save_one(out_dir, sample_name, shape_name=None, n_points=100_000, noise_std=0.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shape_name, points, volume = generate_one(
        shape_name=shape_name,
        n_points=n_points,
        noise_std=noise_std,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_point_cloud(out_dir / f"{sample_name}.ply", pcd)

    with open(out_dir / f"{sample_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"{volume:.12f}\n")

    print(f"{sample_name:20s} | {shape_name:15s} | volume = {volume:.6f} m^3")


def generate_many(out_dir="pcd_samples", n_per_shape=3, n_points=100_000, noise_std=0.0):
    idx = 0
    for shape_name in SHAPES.keys():
        for _ in range(n_per_shape):
            save_one(
                out_dir=out_dir,
                sample_name=f"sample_{idx:02d}",
                shape_name=shape_name,
                n_points=n_points,
                noise_std=noise_std,
            )
            idx += 1


if __name__ == "__main__":
    set_seed(42)

    generate_many(
        out_dir="samples",
        n_per_shape=100,       # shape당 개수
        n_points=random.randint(3000, 6000),    # 밀도
        noise_std=random.uniform(0.001, 0.005)        # 처음엔 0 추천
    )