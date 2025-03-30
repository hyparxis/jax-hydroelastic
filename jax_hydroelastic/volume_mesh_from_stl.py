import jax.numpy as jnp
import numpy as np
import tetgen

from jax_hydroelastic.triangle_mesh_from_stl import triangle_mesh_from_stl
from jax_hydroelastic.volume_mesh import VolumeMesh


def volume_mesh_from_stl(stl_path: str) -> VolumeMesh:
    """Generate a VolumeMesh from an STL surface mesh file using TetGen."""
    tri_mesh = triangle_mesh_from_stl(stl_path)

    # TODO: unecessary copy to numpy array
    vertices = np.array(tri_mesh.vertices)
    elements = np.array(tri_mesh.elements)
    t = tetgen.TetGen(vertices, elements)
    t.tetrahedralize(order=1, mindihedral=20, minratio=1.5)

    return VolumeMesh(jnp.array(t.elem), jnp.array(t.node))
