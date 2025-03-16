import jax
import jax.numpy as jnp

from mesh import Element, Mesh


class Tetrahedron(Element):
    """Represents a tetrahedron as 4 vertex indices."""

    def __init__(self, indices: jax.Array):
        assert indices.size == 4
        assert indices.dtype == jnp.int32

        self.indices = indices


class VolumeMesh(Mesh):
    """Represents a volume mesh as a list of tetrahedra."""

    ElementType = Tetrahedron

    def __init__(self, tetrahedra: jax.Array, vertices: jax.Array):
        assert vertices.shape[1] == 3
        assert len(vertices.shape) == 2
        assert vertices.dtype == jnp.float32

        assert tetrahedra.shape[1] == 4
        assert len(tetrahedra.shape) == 2
        assert tetrahedra.dtype == jnp.int32

        self.elements = tetrahedra
        self.vertices = vertices


indices = jnp.array([1, 2, 3, 4]).reshape(1, 4)
tet = Tetrahedron(indices[0])
verts = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

mesh = VolumeMesh(indices, verts)
mesh.print()

matrix = jnp.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

mesh.transform(
    matrix,
    jnp.zeros(
        3,
    ),
)
mesh.print()
