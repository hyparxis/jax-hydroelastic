import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyvista as pv
from jaxtyping import install_import_hook

with install_import_hook("jax_hydroelastic", "beartype.beartype"):
    from jax_hydroelastic.mesh_intersection import (
        clip_triangle_by_tetrahedron,
        triangulate_polygon,
    )
    from jax_hydroelastic.pyvista_utils import (
        triangle_mesh_to_polydata,
        volume_mesh_to_unstructured_grid,
    )
    from jax_hydroelastic.triangle_mesh import TriangleMesh
    from jax_hydroelastic.volume_mesh import VolumeMesh


def main():
    tet_vertices = jnp.array(
        [
            [2.0, 0.0, 2.0],  # v0
            [-2.0, 0.0, 2.0],  # v1
            [0.0, 2.0, -2.0],  # v2
            [0.0, -2.0, -2.0],  # v3
        ]
    )
    tet_elements = jnp.array([0, 1, 2, 3]).reshape(1, 4)
    volume_mesh = VolumeMesh(tet_elements, tet_vertices)

    tri_vertices = jnp.array(
        [
            [1.5, 1.5, 0.0],  # v1: inside
            [-1.5, 0.0, 0.0],  # v2: outside x=0 and z=0 planes
            [0, -1.5, 0.0],  # v3: outside y=0 and x+y+z=1 planes
        ]
    )
    tri_elements = jnp.array([0, 1, 2]).reshape(1, 3)
    triangle_mesh = TriangleMesh(tri_elements, tri_vertices)

    intersection_polygon = clip_triangle_by_tetrahedron(
        triangle_mesh,
        volume_mesh,
        0,  # triangle index
        0,  # tetrahedron index
        jaxlie.SE3.identity(),
    )

    print("intersection polygon:", intersection_polygon)

    plotter = pv.Plotter()

    # Plot the tetrahedron
    tetrahedron_grid = volume_mesh_to_unstructured_grid(volume_mesh)
    plotter.add_mesh(tetrahedron_grid, style="wireframe")

    # Plot the triangle
    triangle_polydata = triangle_mesh_to_polydata(triangle_mesh)
    plotter.add_mesh(triangle_polydata, style="wireframe", show_edges=True)

    # Plot the intersection polygon surface
    if len(intersection_polygon) >= 3:
        poly_np = np.array(intersection_polygon)
        N = len(poly_np)
        faces = [N] + list(range(N))
        intersection_polydata = pv.PolyData(poly_np, faces)
        plotter.add_mesh(intersection_polydata, color="green")

    # Plot the triangulated intersection mesh
    intersection_triangles, intersection_vertices = triangulate_polygon(
        intersection_polygon, triangle_mesh.face_normal(0)
    )

    intersection_mesh = TriangleMesh(
        jnp.stack(intersection_triangles), jnp.stack(intersection_vertices)
    )
    intersection_polydata = triangle_mesh_to_polydata(intersection_mesh)
    plotter.add_mesh(intersection_polydata, style="wireframe", show_edges=True)

    # Plot the normals
    face_centroids = jax.vmap(
        lambda i0, i1, i2: (
            intersection_mesh.vertices[i0]
            + intersection_mesh.vertices[i1]
            + intersection_mesh.vertices[i2]
        )
        / 3
    )(
        intersection_mesh.elements[:, 0],
        intersection_mesh.elements[:, 1],
        intersection_mesh.elements[:, 2],
    )

    plotter.add_arrows(
        np.array(face_centroids),
        np.array(intersection_mesh.face_normals),
        mag=0.5,
        color="darkgreen",
    )

    plotter.show()


if __name__ == "__main__":
    main()
