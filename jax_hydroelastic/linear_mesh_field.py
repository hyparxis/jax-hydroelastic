from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jax_hydroelastic.mesh import Mesh


class LinearMeshField:
    def __init__(
        self,
        mesh: Mesh,
        values: Float[Array, "num_elements num_vertices_per_element"],
        gradients: Optional[Float[Array, "num_elements 3"]] = None,
    ):
        assert values.shape == (mesh.num_vertices,)
        assert values.dtype == jax.numpy.float32

        self.mesh = mesh
        self.values = values

        if gradients is not None:
            assert gradients.shape == (
                mesh.num_elements(),
                mesh.ElementType.num_vertices(),
            )
            self.gradients = gradients
        else:
            self.gradients = jax.vmap(self._compute_gradient_vector)(
                jnp.arange(mesh.num_elements())
            )

        self.values_at_origin = jax.vmap(self._compute_value_at_origin)(
            jnp.arange(mesh.num_elements())
        )

    def value_at_vertex(self, vertex_index: int) -> jax.Array:
        return self.values[vertex_index]

    def value_at_barycentric_point(
        self,
        element_index: int,
        barycentric_coordinates: Float[Array, "num_vertices_per_element"],
    ) -> float:
        indices = self.mesh.elements[element_index]
        return jnp.sum(self.values[indices] * barycentric_coordinates)

    def value_at_cartesian_point(
        self, element_index: int, point: Float[Array, "3"]
    ) -> float:
        return (
            self.gradients[element_index].dot(point)
            + self.values_at_origin[element_index]
        )

    def gradient_at_element(self, element_index: int) -> Float[Array, "3"]:
        return self.gradients[element_index]

    def _compute_value_at_origin(
        self, element_index: int
    ) -> Float[Array, "num_vertices_per_element"]:
        indices = self.mesh.elements[element_index]
        v0_index = indices[0]
        v0 = self.mesh.vertices[v0_index]
        return self.values[v0_index] - self.gradients[element_index].dot(v0)

    def _compute_gradient_vector(self, element_index: int) -> Float[Array, "3"]:
        indices = self.mesh.elements[element_index]
        values = self.values[indices]
        return self.mesh.gradient_vector_of_linear_field(values, element_index)
