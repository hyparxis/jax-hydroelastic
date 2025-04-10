import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jax_hydroelastic.mesh import Element, Mesh


class Tetrahedron(Element):
    """Represents a tetrahedron as 4 vertex indices."""

    def __init__(self, indices: jax.Array):
        assert indices.size == 4
        assert indices.dtype == jnp.int32

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
        assert vertices.shape[1] == 3
        assert len(vertices.shape) == 2
        assert vertices.dtype == jnp.float32

        assert tetrahedra.shape[1] == 4
        assert len(tetrahedra.shape) == 2
        assert tetrahedra.dtype == jnp.int32

        self.elements = tetrahedra
        self.vertices = vertices

    def gradient_vector_of_linear_field(
        self, field_value: Float[Array, "4"], element_index: int
    ) -> jax.Array:
        pass
