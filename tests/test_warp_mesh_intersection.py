import numpy as np
import pytest

wp = pytest.importorskip("warp")

wp.config.kernel_cache_dir = "/private/tmp/warp-kernel-cache"

from warp_hydroelastic.linear_mesh_field import LinearMeshField
from warp_hydroelastic.mesh_intersection import (
    TriangleBvh,
    TetrahedronBvh,
    clip_triangle_by_tetrahedron,
    sample_volume_field_on_surface,
    sample_volume_field_on_surface_bvh,
    sample_volume_field_on_surface_triangle_bvh,
)
from warp_hydroelastic.triangle_mesh import TriangleMesh
from warp_hydroelastic.volume_mesh import VolumeMesh


def test_clip_triangle_by_tetrahedron_inside_triangle():
    tet_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    tet_elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
    volume_mesh = VolumeMesh.create(tet_elements, tet_vertices, device="cpu")

    tri_vertices = np.array(
        [
            [0.1, 0.1, 0.25],
            [0.5, 0.1, 0.25],
            [0.1, 0.5, 0.25],
        ],
        dtype=np.float32,
    )
    tri_elements = np.array([[0, 1, 2]], dtype=np.int32)
    triangle_mesh = TriangleMesh.create(tri_elements, tri_vertices, device="cpu")

    polygon = clip_triangle_by_tetrahedron(triangle_mesh, volume_mesh, 0, 0)

    assert polygon.num_vertices() == 3
    np.testing.assert_allclose(polygon.numpy(), tri_vertices, atol=1e-6)


def test_sample_volume_field_on_surface_inside_triangle():
    tet_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    volume_mesh = VolumeMesh.create(
        np.array([[0, 1, 2, 3]], dtype=np.int32),
        tet_vertices,
        device="cpu",
    )

    triangle_mesh = TriangleMesh.create(
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array(
            [
                [0.1, 0.1, 0.25],
                [0.5, 0.1, 0.25],
                [0.1, 0.5, 0.25],
            ],
            dtype=np.float32,
        ),
        device="cpu",
    )

    field = LinearMeshField.create(
        volume_mesh,
        np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    )

    surface = sample_volume_field_on_surface(triangle_mesh, field)

    assert surface.polygon_sizes.numpy()[0, 0] == 3
    assert surface.surface_sizes.numpy()[0, 0] == 3

    pressures = surface.surface_pressures.numpy()[0, 0, :4]
    np.testing.assert_allclose(
        pressures,
        np.array([1.05, 1.45, 1.85, 1.45], dtype=np.float32),
        atol=1e-5,
    )


def test_sample_volume_field_on_surface_bvh_matches_brute_force():
    tet_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 2.0],
            [2.0, 3.0, 2.0],
            [2.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    volume_mesh = VolumeMesh.create(
        np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32),
        tet_vertices,
        device="cpu",
    )

    triangle_mesh = TriangleMesh.create(
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array(
            [
                [0.1, 0.1, 0.25],
                [0.5, 0.1, 0.25],
                [0.1, 0.5, 0.25],
            ],
            dtype=np.float32,
        ),
        device="cpu",
    )

    field = LinearMeshField.create(
        volume_mesh,
        np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    )
    tetrahedron_bvh = TetrahedronBvh.create(volume_mesh)

    brute_force = sample_volume_field_on_surface(triangle_mesh, field)
    accelerated = sample_volume_field_on_surface_bvh(
        triangle_mesh,
        field,
        tetrahedron_bvh=tetrahedron_bvh,
    )

    np.testing.assert_array_equal(
        accelerated.polygon_sizes.numpy(),
        brute_force.polygon_sizes.numpy(),
    )
    np.testing.assert_array_equal(
        accelerated.surface_sizes.numpy(),
        brute_force.surface_sizes.numpy(),
    )
    np.testing.assert_allclose(
        accelerated.surface_pressures.numpy(),
        brute_force.surface_pressures.numpy(),
        atol=1e-5,
    )
    assert accelerated.surface_sizes.numpy()[1, 0] == 0


def test_sample_volume_field_on_surface_triangle_bvh_matches_brute_force():
    tet_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 2.0],
            [2.0, 3.0, 2.0],
            [2.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    volume_mesh = VolumeMesh.create(
        np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32),
        tet_vertices,
        device="cpu",
    )

    triangle_mesh = TriangleMesh.create(
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array(
            [
                [0.1, 0.1, 0.25],
                [0.5, 0.1, 0.25],
                [0.1, 0.5, 0.25],
            ],
            dtype=np.float32,
        ),
        device="cpu",
    )

    field = LinearMeshField.create(
        volume_mesh,
        np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    )
    triangle_bvh = TriangleBvh.create(triangle_mesh)

    brute_force = sample_volume_field_on_surface(triangle_mesh, field)
    accelerated = sample_volume_field_on_surface_triangle_bvh(
        triangle_mesh,
        field,
        triangle_bvh=triangle_bvh,
    )

    np.testing.assert_array_equal(
        accelerated.polygon_sizes.numpy(),
        brute_force.polygon_sizes.numpy(),
    )
    np.testing.assert_array_equal(
        accelerated.surface_sizes.numpy(),
        brute_force.surface_sizes.numpy(),
    )
    np.testing.assert_allclose(
        accelerated.surface_pressures.numpy(),
        brute_force.surface_pressures.numpy(),
        atol=1e-5,
    )
    assert accelerated.surface_sizes.numpy()[1, 0] == 0
