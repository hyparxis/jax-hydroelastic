import argparse
import os
import sys
import statistics
import time
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from warp_hydroelastic.linear_mesh_field import LinearMeshField
from warp_hydroelastic.mesh_intersection import (
    TetrahedronBvh,
    TriangleBvh,
    sample_volume_field_on_surface,
    sample_volume_field_on_surface_bvh,
    sample_volume_field_on_surface_triangle_bvh,
)
from warp_hydroelastic.triangle_mesh import TriangleMesh
from warp_hydroelastic.volume_mesh import VolumeMesh


def _default_device() -> str:
    if wp.is_cuda_available():
        return "cuda:0"
    return "cpu"


def _make_tetrahedron_grid(
    n: int,
    spacing: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = []
    elements = []
    values = []

    for k in range(n):
        for j in range(n):
            for i in range(n):
                origin = np.array([i * spacing, j * spacing, k * spacing], dtype=np.float32)
                base = len(vertices)
                tet_vertices = np.array(
                    [
                        origin,
                        origin + np.array([scale, 0.0, 0.0], dtype=np.float32),
                        origin + np.array([0.0, scale, 0.0], dtype=np.float32),
                        origin + np.array([0.0, 0.0, scale], dtype=np.float32),
                    ],
                    dtype=np.float32,
                )

                vertices.extend(tet_vertices)
                elements.append([base, base + 1, base + 2, base + 3])
                values.extend(tet_vertices[:, 0] + 2.0 * tet_vertices[:, 1] + 3.0 * tet_vertices[:, 2])

    return (
        np.asarray(elements, dtype=np.int32),
        np.asarray(vertices, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
    )


def _make_plane_triangle_grid(
    n: int,
    extent: float,
    z: float,
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, extent, n + 1, dtype=np.float32)
    ys = np.linspace(0.0, extent, n + 1, dtype=np.float32)

    vertices = []
    for y in ys:
        for x in xs:
            vertices.append([x, y, z])

    elements = []
    for j in range(n):
        for i in range(n):
            v00 = j * (n + 1) + i
            v10 = v00 + 1
            v01 = v00 + (n + 1)
            v11 = v01 + 1
            elements.append([v00, v10, v11])
            elements.append([v00, v11, v01])

    return np.asarray(elements, dtype=np.int32), np.asarray(vertices, dtype=np.float32)


def make_scene(
    tet_grid: int,
    tri_grid: int,
    device: str,
    spacing: float = 1.0,
    tet_scale: float = 0.9,
    plane_z: float = 0.25,
) -> tuple[TriangleMesh, LinearMeshField]:
    tet_elements, tet_vertices, pressures = _make_tetrahedron_grid(
        tet_grid,
        spacing,
        tet_scale,
    )
    tri_elements, tri_vertices = _make_plane_triangle_grid(
        tri_grid,
        extent=tet_grid * spacing,
        z=plane_z,
    )

    volume_mesh = VolumeMesh.create(tet_elements, tet_vertices, device=device)
    triangle_mesh = TriangleMesh.create(tri_elements, tri_vertices, device=device)
    field = LinearMeshField.create(volume_mesh, pressures)
    wp.synchronize()
    return triangle_mesh, field


def time_call(name: str, fn, iterations: int, warmup: int) -> dict[str, float | int | str]:
    result = None
    for _ in range(warmup):
        result = fn()
        wp.synchronize()

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        wp.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    assert result is not None
    sizes = result.surface_sizes.numpy()
    intersections = int(np.count_nonzero(sizes))

    return {
        "name": name,
        "min_ms": min(timings),
        "median_ms": statistics.median(timings),
        "max_ms": max(timings),
        "intersections": intersections,
    }


def time_build(name: str, fn, iterations: int) -> tuple[object, dict[str, float | str]]:
    result = None
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        wp.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    assert result is not None
    return result, {
        "name": name,
        "min_ms": min(timings),
        "median_ms": statistics.median(timings),
        "max_ms": max(timings),
    }


def print_table(title: str, rows: list[dict[str, float | int | str]]) -> None:
    print()
    print(title)
    print("-" * len(title))
    if not rows:
        return

    has_intersections = "intersections" in rows[0]
    if has_intersections:
        print(f"{'design':32s} {'min ms':>10s} {'median ms':>10s} {'max ms':>10s} {'hits':>10s}")
        for row in rows:
            print(
                f"{row['name']:32s} {row['min_ms']:10.3f} {row['median_ms']:10.3f} "
                f"{row['max_ms']:10.3f} {row['intersections']:10d}"
            )
    else:
        print(f"{'structure':32s} {'min ms':>10s} {'median ms':>10s} {'max ms':>10s}")
        for row in rows:
            print(
                f"{row['name']:32s} {row['min_ms']:10.3f} {row['median_ms']:10.3f} "
                f"{row['max_ms']:10.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tet-grid", type=int, default=8)
    parser.add_argument("--tri-grid", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--build-iterations", type=int, default=3)
    parser.add_argument("--skip-brute-force", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    wp.config.kernel_cache_dir = os.environ.get(
        "WARP_KERNEL_CACHE_DIR",
        "/private/tmp/warp-kernel-cache",
    )
    wp.init()

    device = _default_device() if args.device == "auto" else args.device
    triangle_mesh, field = make_scene(args.tet_grid, args.tri_grid, device)

    num_tets = field.mesh.num_elements()
    num_triangles = triangle_mesh.num_elements()
    print(f"device: {device}")
    print(f"tets: {num_tets}")
    print(f"triangles: {num_triangles}")
    print(f"dense pairs: {num_tets * num_triangles}")

    tetrahedron_bvh, tet_bvh_stats = time_build(
        "tetrahedron AABB BVH",
        lambda: TetrahedronBvh.create(field.mesh),
        args.build_iterations,
    )
    triangle_bvh, tri_bvh_stats = time_build(
        "triangle AABB BVH",
        lambda: TriangleBvh.create(triangle_mesh),
        args.build_iterations,
    )

    print_table("Build Times", [tet_bvh_stats, tri_bvh_stats])

    rows = []
    brute_force = None
    if not args.skip_brute_force:
        rows.append(
            time_call(
                "brute force pair-parallel",
                lambda: sample_volume_field_on_surface(triangle_mesh, field),
                args.iterations,
                args.warmup,
            )
        )
        if args.check:
            brute_force = sample_volume_field_on_surface(triangle_mesh, field)
            wp.synchronize()

    rows.append(
        time_call(
            "triangle -> tet BVH",
            lambda: sample_volume_field_on_surface_bvh(
                triangle_mesh,
                field,
                tetrahedron_bvh=tetrahedron_bvh,
            ),
            args.iterations,
            args.warmup,
        )
    )
    rows.append(
        time_call(
            "tet -> triangle BVH",
            lambda: sample_volume_field_on_surface_triangle_bvh(
                triangle_mesh,
                field,
                triangle_bvh=triangle_bvh,
            ),
            args.iterations,
            args.warmup,
        )
    )

    print_table("Contact Sampling Times", rows)

    if args.check and brute_force is not None:
        tet_bvh_surface = sample_volume_field_on_surface_bvh(
            triangle_mesh,
            field,
            tetrahedron_bvh=tetrahedron_bvh,
        )
        tri_bvh_surface = sample_volume_field_on_surface_triangle_bvh(
            triangle_mesh,
            field,
            triangle_bvh=triangle_bvh,
        )
        wp.synchronize()
        np.testing.assert_array_equal(
            tet_bvh_surface.surface_sizes.numpy(),
            brute_force.surface_sizes.numpy(),
        )
        np.testing.assert_array_equal(
            tri_bvh_surface.surface_sizes.numpy(),
            brute_force.surface_sizes.numpy(),
        )
        print()
        print("Correctness check: BVH surface sizes match brute force.")


if __name__ == "__main__":
    main()
