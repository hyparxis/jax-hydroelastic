import jax.numpy as jnp
import numpy as np
from stl import mesh

from triangle_mesh import TriangleMesh


def triangle_mesh_from_stl(stl_path: str) -> TriangleMesh:
    """Create a TriangleMesh from an STL file."""

    stl_mesh = mesh.Mesh.from_file(stl_path)

    all_vertices = stl_mesh.vectors.reshape(-1, 3)
    unique_vertices, unique_indices = np.unique(
        all_vertices, axis=0, return_inverse=True
    )

    triangle_index_rows = unique_indices.reshape(-1, 3)

    indices_jax = jnp.array(triangle_index_rows, dtype=jnp.int32)
    vertices_jax = jnp.array(unique_vertices, dtype=jnp.float32)

    return TriangleMesh(indices_jax, vertices_jax)
