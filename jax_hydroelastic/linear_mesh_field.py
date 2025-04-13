from typing import Optional

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jax_hydroelastic.mesh import Mesh


@chex.dataclass
class LinearMeshField:
    mesh: Mesh
    values: Float[Array, "num_vertices"]
    gradients: Float[Array, "num_elements 3"]
    values_at_origin: Float[Array, "num_elements"]

    @classmethod
    def create(
        cls,
        mesh: Mesh,
        values: Float[Array, "num_vertices"],
        gradients: Optional[Float[Array, "num_elements 3"]] = None,
    ):
        if gradients is None:
            gradients = jax.vmap(cls._compute_gradient_vector, in_axes=(None, None, 0))(
                mesh, values, jnp.arange(mesh.num_elements())
            )

        values_at_origin = jax.vmap(
            cls._compute_value_at_origin, in_axes=(None, None, None, 0)
        )(mesh, values, gradients, jnp.arange(mesh.num_elements()))
        return cls(
            mesh=mesh,
            values=values,
            gradients=gradients,
            values_at_origin=values_at_origin,
        )

    def value_at_vertex(self, vertex_index: int | Int[Array, ""]) -> jax.Array:
        return self.values[vertex_index]

    def value_at_barycentric_point(
        self,
        element_index: int | Int[Array, ""],
        barycentric_coordinates: Float[Array, "num_vertices_per_element"],
    ) -> Float[Array, ""]:
        indices = self.mesh.elements[element_index]
        return jnp.sum(self.values[indices] * barycentric_coordinates)

    def value_at_cartesian_point(
        self, element_index: int | Int[Array, ""], point: Float[Array, "3"]
    ) -> Float[Array, ""]:
        return (
            self.gradients[element_index].dot(point)
            + self.values_at_origin[element_index]
        )

    def gradient_at_element(
        self, element_index: int | Int[Array, ""]
    ) -> Float[Array, "3"]:
        return self.gradients[element_index]

    @staticmethod
    def _compute_value_at_origin(
        mesh: Mesh,
        values: Float[Array, "num_vertices"],
        gradients: Float[Array, "num_elements 3"],
        element_index: int | Int[Array, ""],
    ) -> Float[Array, ""]:
        indices = mesh.elements[element_index]
        v0_index = indices[0]
        v0 = mesh.vertices[v0_index]
        return values[v0_index] - gradients[element_index].dot(v0)

    @staticmethod
    def _compute_gradient_vector(
        mesh: Mesh,
        values: Float[Array, "num_vertices"],
        element_index: int | Int[Array, ""],
    ) -> Float[Array, "3"]:
        indices = mesh.elements[element_index]
        element_values = values[indices]
        return mesh.gradient_vector_of_linear_field(element_values, element_index)
