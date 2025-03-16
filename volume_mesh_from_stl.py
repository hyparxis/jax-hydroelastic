import jax.numpy as jnp
import numpy as np
import tetgen

from triangle_mesh_from_stl import triangle_mesh_from_stl
from volume_mesh import VolumeMesh


def volume_mesh_from_stl(stl_path: str) -> VolumeMesh:
    """Create a VolumeMesh from an STL file."""
    tri_mesh = triangle_mesh_from_stl(stl_path)

    vertices = np.array(tri_mesh.vertices)
    elements = np.array(tri_mesh.elements)
    t = tetgen.TetGen(vertices, elements)
    t.tetrahedralize()

    return VolumeMesh(jnp.array(t.elem), jnp.array(t.node))
