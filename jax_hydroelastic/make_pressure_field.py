import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

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

    # Extract unique faces and their counts
    unique_faces, counts = jnp.unique(face_indices_sorted, axis=0, return_counts=True)

    # Keep only boundary faces (i.e. faces that belong to only one tetrahedron)
    boundary_faces = unique_faces[counts == 1]

    return TriangleMesh(
        triangles=boundary_faces,
        vertices=vertices,
    )


def point_to_triangle_distance(
    p: Float[Array, "3"],
    A: Float[Array, "3"],
    B: Float[Array, "3"],
    C: Float[Array, "3"],
) -> Float[Array, ""]:
    v0 = B - A
    v1 = C - A
    v2 = p - A

    d00 = jnp.dot(v0, v0)
    d01 = jnp.dot(v0, v1)
    d11 = jnp.dot(v1, v1)
    d02 = jnp.dot(v0, v2)
    d12 = jnp.dot(v1, v2)

    denom = d00 * d11 - d01 * d01
    denom = jnp.maximum(denom, 1e-14)

    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom

    u_clamped = jnp.clip(u, 0.0, 1.0)
    v_clamped = jnp.clip(v, 0.0, 1.0)

    s = u_clamped + v_clamped

    over = jnp.maximum(s - 1.0, 0.0)
    u_clamped -= over * (u_clamped / (s + 1e-14))
    v_clamped -= over * (v_clamped / (s + 1e-14))

    c = A + u_clamped * v0 + v_clamped * v1

    return jnp.linalg.norm(p - c)


def point_to_surface_distance(
    p: Float[Array, "3"], vertices: Float[Array, "V 3"], triangles: Int[Array, "T 3"]
) -> Float[Array, ""]:
    tri_verts = vertices[triangles]  # shape (T, 3, 3)
    distances = jax.vmap(lambda tri: point_to_triangle_distance(p, *tri))(tri_verts)
    return jnp.min(distances)


def points_to_surface_distance(
    points: Float[Array, "N 3"],
    vertices: Float[Array, "V 3"],
    triangles: Int[Array, "T 3"],
) -> Float[Array, "N"]:
    return jax.vmap(lambda p: point_to_surface_distance(p, vertices, triangles))(points)


def make_pressure_field(volume_mesh: VolumeMesh, hydroelastic_modulus: float):
    surface_mesh = volume_mesh_to_surface_mesh(volume_mesh)
    distances = points_to_surface_distance(
        volume_mesh.vertices, surface_mesh.vertices, surface_mesh.elements
    )
    max_distance = jnp.max(distances)
    pressures = hydroelastic_modulus * distances / max_distance
    return pressures
