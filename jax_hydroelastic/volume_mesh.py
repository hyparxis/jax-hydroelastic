import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jax_hydroelastic.mesh import Mesh


@chex.dataclass
class VolumeMesh(Mesh):
    """Represents a volume mesh as a list of tetrahedra."""

    elements: Int[Array, "num_elements 4"]

    @classmethod
    def create(
        cls,
        tetrahedra: Int[Array, "num_elements 4"],
        vertices: Float[Array, "num_vertices 3"],
    ):
        """Trivial class method to keep a consistent interface with TriangleMesh."""
        return cls(
            elements=tetrahedra,
            vertices=vertices,
        )

    @classmethod
    def num_vertices_per_element(cls) -> int:
        return 4

    def gradient_vector_of_linear_field(
        self, field_value: Float[Array, "4"], element_index: int | Int[Array, ""]
    ) -> jax.Array:
        g0 = field_value[0] * self._barycentric_gradient(element_index, 0)
        g1 = field_value[1] * self._barycentric_gradient(element_index, 1)
        g2 = field_value[2] * self._barycentric_gradient(element_index, 2)
        g3 = field_value[3] * self._barycentric_gradient(element_index, 3)
        return g0 + g1 + g2 + g3

    def _barycentric_gradient(
        self, e: int | Int[Array, ""], v: int | Int[Array, ""]
    ) -> Float[Array, "3"]:
        V = self.vertices[self.elements[e, v]]
        A = self.vertices[self.elements[e, (v + 1) % 4]]
        B = self.vertices[self.elements[e, (v + 2) % 4]]
        C = self.vertices[self.elements[e, (v + 3) % 4]]

        AV = V - A
        AB = B - A
        AC = C - A

        area_vector = jnp.cross(AB, AC)
        signed_volume = jnp.dot(area_vector, AV)

        return area_vector / signed_volume
