from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jax_hydroelastic.mesh import Mesh


class TriangleMesh(Mesh):
    """Represents a triangle mesh as a list of triangles."""

    def __init__(
        self,
        triangles: Int[Array, "num_elements 3"],
        vertices: Float[Array, "num_vertices 3"],
        face_normals: Optional[Float[Array, "num_elements 3"]] = None,
    ):
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
            self.face_normals = face_normals

    @classmethod
    def num_vertices_per_element(cls) -> int:
        return 3

    def face_normal(self, index: int | Int[Array, ""]) -> Float[Array, "3"]:
        """Get the normal vector of a face."""
        return self.face_normals[index]

    def gradient_vector_of_linear_field(
        self, field_value: Float[Array, "3"], element_index: Int[Array, ""]
    ) -> Float[Array, "3"]:
        g0 = field_value[0] * self._barycentric_gradient(element_index, 0)
        g1 = field_value[1] * self._barycentric_gradient(element_index, 1)
        g2 = field_value[2] * self._barycentric_gradient(element_index, 2)
        return g0 + g1 + g2

    def _barycentric_gradient(
        self, e: int | Int[Array, ""], v: int | Int[Array, ""]
    ) -> Float[Array, "3"]:
        # TODO: handle small area case
        v = self.vertices[self.elements[e, v]]
        a = self.vertices[self.elements[e, (v + 1) % 3]]
        b = self.vertices[self.elements[e, (v + 2) % 3]]

        # AB = B - A
        ab = b - a
        # AV = V - A
        av = v - a

        # AV - ( (AV · AB) / |AB|^2 ) * AB
        return av - jnp.dot(av, ab) / jnp.dot(ab, ab) * ab
