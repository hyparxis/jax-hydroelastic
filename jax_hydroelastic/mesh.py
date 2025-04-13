from abc import ABC, abstractmethod

import chex
import jax
import jax.numpy as jnp
import jaxlie
from jaxtyping import Array, Float, Int


def reverse_triangle_winding(indices: Int[Array, "3"]) -> Int[Array, "3"]:
    """Reverse the winding order of a triangle."""
    return jnp.array([indices[1], indices[0], indices[2]])


def flip_tetrahedron_orientation(indices: Int[Array, "4"]) -> Int[Array, "4"]:
    """Flip the orientation of a tetrahedron."""
    return jnp.array([indices[1], indices[0], indices[2], indices[3]])


@chex.dataclass
class Mesh(ABC):
    elements: Int[Array, "num_elements num_vertices_per_element"]
    vertices: Float[Array, "num_vertices 3"]

    @classmethod
    @abstractmethod
    def num_vertices_per_element(cls) -> int:
        """Return the number of vertices per element."""
        pass

    @abstractmethod
    def gradient_vector_of_linear_field(
        self, field_value: Float[Array, "num_vertices_per_element"], element_index: int
    ) -> jax.Array:
        """Compute the gradient vector of a linear field defined on the mesh."""
        pass

    def num_elements(self) -> int:
        return self.elements.shape[0]

    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    def transform(self, transformation: jaxlie.SE3) -> None:
        # TODO: check this batch matmul works with jaxlie
        return self.replace(vertices=(transformation @ self.vertices))

    def print(self) -> None:
        print("elements: ")
        for element in self.elements:
            print(element)
        print("vertices: ")
        print(self.vertices)
