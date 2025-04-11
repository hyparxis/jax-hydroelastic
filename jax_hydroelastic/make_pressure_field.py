import jax.numpy as jnp

from jax_hydroelastic.triangle_mesh import TriangleMesh
from jax_hydroelastic.volume_mesh import VolumeMesh


def volume_mesh_to_surface_mesh(volume_mesh: VolumeMesh) -> TriangleMesh:
    tetrahedra = volume_mesh.elements
    vertices = volume_mesh.vertices

    # Enumerate all 4 faces of each tetrahedron
    face_indices = jnp.vstack(
        [
            tetrahedra[:, [0, 1, 2]],
            tetrahedra[:, [0, 1, 3]],
            tetrahedra[:, [0, 2, 3]],
            tetrahedra[:, [1, 2, 3]],
        ]
    )  # (4 * num_tetrahedra, 3)

    # Canonicalize each face by sorting their vertex indices
    face_indices_sorted = jnp.sort(face_indices, axis=1)

    # unique_faces will be shape (k, 3) where k <= 4 * num_tetrahedra
    # counts will be shape (k,)
    unique_faces, counts = jnp.unique(face_indices_sorted, axis=0, return_counts=True)

    # Keep only faces with count == 1 (i.e. boundary faces)
    boundary_faces = unique_faces[counts == 1]

    return TriangleMesh(
        triangles=boundary_faces,
        vertices=vertices,
    )


# def make_pressure_field(volume_mesh: VolumeMesh, hydroelastic_modulus: float):
