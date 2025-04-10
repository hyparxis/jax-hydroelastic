from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jax_hydroelastic.mesh import Element, Mesh


class Triangle(Element):
    """Represents a triangle as 3 vertex indices."""

    def __init__(self, indices: jax.Array):
        assert indices.size == 3
        assert indices.dtype == jnp.int32

        self.indices = indices

    @classmethod
    def num_vertices(self) -> int:
        return 3

    def reverse_winding(self) -> None:
        """Reverse the winding order of the triangle."""
        self.flip_orientation()


class TriangleMesh(Mesh):
    """Represents a triangle mesh as a list of triangles."""

    ElementType = Triangle

    def __init__(
        self,
        triangles: Int[Array, "num_elements 3"],
        vertices: Float[Array, "num_vertices 3"],
        face_normals: Optional[Float[Array, "num_elements 3"]] = None,
    ):
        assert vertices.shape[1] == 3
        assert len(vertices.shape) == 2
        assert vertices.dtype == jnp.float32

        assert triangles.shape[1] == 3
        assert len(triangles.shape) == 2
        assert triangles.dtype == jnp.int32

        self.elements = triangles
        self.vertices = vertices

        if face_normals is None:

            def compute_face_normal(
                v0: Float[Array, "3"], v1: Float[Array, "3"], v2: Float[Array, "3"]
            ) -> Float[Array, "3"]:
                face_normal = jnp.cross(v1 - v0, v2 - v0)
                norm = jnp.linalg.norm(face_normal)
                norm_safe = jnp.maximum(norm, 1e-14)
                return face_normal / norm_safe

            self.face_normals = jax.vmap(
                compute_face_normal,
                in_axes=(0, 0, 0),
            )(
                vertices[triangles[:, 0]],
                vertices[triangles[:, 1]],
                vertices[triangles[:, 2]],
            )
        else:
            assert face_normals.shape == triangles.shape
            assert face_normals.dtype == jnp.float32
            self.face_normals = face_normals

    def face_normal(self, index: int) -> jax.Array:
        """Get the normal vector of a face."""
        return self.face_normals[index]

    def gradient_vector_of_linear_field(
        self, field_value: Float[Array, "3"], element_index: int
    ) -> jax.Array:
        pass
