import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jax_hydroelastic.mesh import Element, Mesh


class Tetrahedron(Element):
    """Represents a tetrahedron as 4 vertex indices."""

    def __init__(self, indices: Int[Array, "4"]):
        self.indices = indices

    @classmethod
    def num_vertices(self) -> int:
        return 4


class VolumeMesh(Mesh):
    """Represents a volume mesh as a list of tetrahedra."""

    ElementType = Tetrahedron

    def __init__(
        self,
        tetrahedra: Int[Array, "num_elements 4"],
        vertices: Float[Array, "num_vertices 3"],
    ):
        self.elements = tetrahedra
        self.vertices = vertices

    def gradient_vector_of_linear_field(
        self, field_value: Float[Array, "4"], element_index: int
    ) -> jax.Array:
        g0 = field_value[0] * self._barycentric_gradient(element_index, 0)
        g1 = field_value[1] * self._barycentric_gradient(element_index, 1)
        g2 = field_value[2] * self._barycentric_gradient(element_index, 2)
        g3 = field_value[3] * self._barycentric_gradient(element_index, 3)
        return g0 + g1 + g2 + g3

    def _barycentric_gradient(
        self, e: int | Int[Array, ""], v: int | Int[Array, ""]
    ) -> Float[Array, "3"]:
        v = self.vertices[self.triangles[e, v]]
        a = self.vertices[self.triangles[e, (v + 1) % 4]]
        b = self.vertices[self.triangles[e, (v + 2) % 4]]
        c = self.vertices[self.triangles[e, (v + 3) % 4]]

        ab = a - b
        ac = a - c
        av = a - v

        area_vector = jnp.cross(ab, ac)
        signed_volume = jnp.dot(area_vector, av)

        return area_vector / signed_volume
