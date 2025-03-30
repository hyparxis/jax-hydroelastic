import jax
import jax.numpy as jnp

from jax_hydroelastic.mesh import Element, Mesh


class Triangle(Element):
    """Represents a triangle as 3 vertex indices."""

    def __init__(self, indices: jax.Array):
        assert indices.size == 3
        assert indices.dtype == jnp.int32

        self.indices = indices

    def reverse_winding(self) -> None:
        """Reverse the winding order of the triangle."""
        self.flip_orientation()


class TriangleMesh(Mesh):
    """Represents a triangle mesh as a list of triangles."""

    ElementType = Triangle

    def __init__(self, triangles: jax.Array, vertices: jax.Array):
        assert vertices.shape[1] == 3
        assert len(vertices.shape) == 2
        assert vertices.dtype == jnp.float32

        assert triangles.shape[1] == 3
        assert len(triangles.shape) == 2
        assert triangles.dtype == jnp.int32

        self.elements = triangles
        self.vertices = vertices
