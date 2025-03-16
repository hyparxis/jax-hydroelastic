import matplotlib.pyplot as plt
import numpy as np

from triangle_mesh import TriangleMesh
from volume_mesh_from_stl import volume_mesh_from_stl


def plot_triangle_mesh(tri_mesh: TriangleMesh):
    """
    Plot the TriangleMesh in 3D as wireframe lines for each triangle.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    vertices = np.array(tri_mesh.vertices)  # shape (N, 3)
    elements = np.array(tri_mesh.elements)  # shape (M, 3)

    # Draw each triangle as a polygon outline
    for tri in elements:
        tri_coords = vertices[tri]  # shape (3, 3)
        xs, ys, zs = tri_coords[:, 0], tri_coords[:, 1], tri_coords[:, 2]

        # Close the loop by repeating the first point
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        zs = np.append(zs, zs[0])

        ax.plot(xs, ys, zs)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def main():
    # mesh = triangle_mesh_from_stl("sphere.stl")
    # plot_triangle_mesh(mesh)
    tet_mesh = volume_mesh_from_stl("sphere.stl")


if __name__ == "__main__":
    main()
