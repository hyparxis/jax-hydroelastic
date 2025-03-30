import numpy as np
import pyvista as pv

from jax_hydroelastic.triangle_mesh import TriangleMesh
from jax_hydroelastic.volume_mesh import VolumeMesh


def triangle_mesh_to_polydata(tri_mesh: TriangleMesh) -> pv.PolyData:
    """
    Convert a TriangleMesh into a pyvista PolyData object.
    """
    # Convert JAX arrays to NumPy.
    vertices = np.array(tri_mesh.vertices)  # shape: (N, 3)
    triangles = np.array(tri_mesh.elements)  # shape: (M, 3)

    # PyVista requires a 'faces' array of the form [3, i0, i1, i2, 3, i3, i4, i5, ...].
    # The leading '3' indicates the number of vertices in each face (triangle).
    faces = np.column_stack([np.full(triangles.shape[0], 3), triangles]).ravel()

    # Create PolyData
    return pv.PolyData(vertices, faces)


def volume_mesh_to_unstructured_grid(vol_mesh: VolumeMesh) -> pv.UnstructuredGrid:
    """
    Convert a VolumeMesh with tetrahedra into pyvista.UnstructuredGrid.
    """
    # Convert from JAX to NumPy
    vertices = np.array(vol_mesh.vertices)  # shape: (N, 3)
    tetrahedra = np.array(vol_mesh.elements)  # shape: (M, 4)

    num_tets = tetrahedra.shape[0]

    # The 'cells' array for an UnstructuredGrid is:
    # [4, i0, i1, i2, i3, 4, i4, i5, i6, i7, ...]
    # where '4' is the number of points in a tetrahedron.
    cells = np.column_stack([np.full(num_tets, 4), tetrahedra]).ravel()

    # Cell types array: each cell is a VTK_TETRA = 10
    cell_types = np.full(num_tets, pv.CellType.TETRA, dtype=np.uint8)

    # Create the UnstructuredGrid
    return pv.UnstructuredGrid(cells, cell_types, vertices)
