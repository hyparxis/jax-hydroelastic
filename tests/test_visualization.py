from jax_hydroelastic.make_pressure_field import volume_mesh_to_surface_mesh
from jax_hydroelastic.pyvista_utils import (
    triangle_mesh_to_polydata,
    volume_mesh_to_unstructured_grid,
)
from jax_hydroelastic.triangle_mesh_from_stl import triangle_mesh_from_stl
from jax_hydroelastic.volume_mesh_from_stl import volume_mesh_from_stl


def visualize_volume_mesh() -> None:
    tet_mesh = volume_mesh_from_stl("tests/assets/sphere.stl")
    grid = volume_mesh_to_unstructured_grid(tet_mesh)

    # get cell centroids
    cells = grid.cells.reshape(-1, 5)[:, 1:]
    cell_center = grid.points[cells].mean(1)

    # extract cells below the 0 xy plane
    mask = cell_center[:, 2] < 0
    cell_ind = mask.nonzero()[0]
    subgrid = grid.extract_cells(cell_ind)
    cell_qual = subgrid.compute_cell_quality()["CellQuality"]
    subgrid.plot(
        scalars=cell_qual,
        cmap="bwr",
        clim=[0, 1],
        flip_scalars=True,
        show_edges=True,
    )


def visualize_triangle_mesh() -> None:
    tri_mesh = triangle_mesh_from_stl("tests/assets/sphere.stl")
    polydata = triangle_mesh_to_polydata(tri_mesh)
    polydata.plot(show_edges=True)


def visualize_surface_mesh() -> None:
    tet_mesh = volume_mesh_from_stl("tests/assets/sphere.stl")
    surface_mesh = volume_mesh_to_surface_mesh(tet_mesh)
    surface_polydata = triangle_mesh_to_polydata(surface_mesh)
    surface_polydata.plot(show_edges=True)


def main():
    visualize_surface_mesh()


if __name__ == "__main__":
    main()
