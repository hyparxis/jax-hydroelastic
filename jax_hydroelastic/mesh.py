from abc import ABC, abstractmethod
from typing import Type

import jax
import jaxlie
from jaxtyping import Array, Float


class Element(ABC):
    def flip_orientation(self) -> None:
        """Flip the orientation of the tetrahedron or triangle."""

        # You can flip the orientation of a tetrahedron or winding order of a triangle
        # by swapping the first two indices.
        self.indices[0], self.indices[1] = self.indices[1], self.indices[0]

    def print(self) -> None:
        print(self.indices)

    @classmethod
    @abstractmethod
    def num_vertices(self) -> int:
        pass


class Mesh(ABC):
    ElementType: Type[Element]

    @abstractmethod
    def gradient_vector_of_linear_field(
        self, field_value: Float[Array, "num_vertices_per_element"], element_index: int
    ) -> jax.Array:
        """Compute the gradient vector of a linear field defined on the mesh."""
        pass

    def num_elements(self) -> int:
        return len(self.elements)

    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    def get_element(self, index: int) -> Element:
        return self.ElementType(self.elements[index])

    def get_vertex(self, index: int) -> jax.Array:
        return self.vertices[index]

    def transform(self, transformation: jaxlie.SE3) -> None:
        # TODO: check this batch matmul works with jaxlie
        self.vertices = transformation @ self.vertices

    def print(self) -> None:
        print("elements: ")
        for element in self.elements:
            self.ElementType(element).print()
        print("vertices: ")
        print(self.vertices)
