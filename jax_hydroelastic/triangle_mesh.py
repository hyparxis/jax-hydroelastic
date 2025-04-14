from typing import Optional

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jax_hydroelastic.mesh import Mesh


@chex.dataclass
class TriangleMesh(Mesh):
    """Represents a triangle mesh as a list of triangles."""

    elements: Int[Array, "num_elements 3"]
    face_normals: Float[Array, "num_elements 3"]

    @classmethod
    def create(
        cls,
        triangles: Int[Array, "num_elements 3"],
        vertices: Float[Array, "num_vertices 3"],
        face_normals: Optional[Float[Array, "num_elements 3"]] = None,
    ):
        if face_normals is None:

            def _compute_face_normal(
                v0: Float[Array, "3"], v1: Float[Array, "3"], v2: Float[Array, "3"]
            ) -> Float[Array, "3"]:
                face_normal = jnp.cross(v1 - v0, v2 - v0)
                norm = jnp.linalg.norm(face_normal)
                norm_safe = jnp.maximum(norm, 1e-14)
                return face_normal / norm_safe

            face_normals = jax.vmap(
                _compute_face_normal,
                in_axes=(0, 0, 0),
            )(
                vertices[triangles[:, 0]],
                vertices[triangles[:, 1]],
                vertices[triangles[:, 2]],
            )
        return cls(
            elements=triangles,
            vertices=vertices,
            face_normals=face_normals,
        )

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
        V = self.vertices[self.elements[e, v]]
        A = self.vertices[self.elements[e, (v + 1) % 3]]
        B = self.vertices[self.elements[e, (v + 2) % 3]]

        AB = B - A
        AV = V - A

        # AV - ( (AV · AB) / |AB|^2 ) * AB
        return AV - jnp.dot(AV, AB) / jnp.dot(AB, AB) * AB
