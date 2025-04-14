from typing import List, Tuple

import chex
import jax
import jax.numpy as jnp
import jaxlie
from jaxtyping import Array, Float, Int

from jax_hydroelastic.triangle_mesh import TriangleMesh
from jax_hydroelastic.volume_mesh import VolumeMesh


@chex.dataclass
class PlaneData:
    """Data for a plane defined by the implicit equation: `P(x⃗) = n̂⋅x⃗ - d = 0`"""

    normal: Float[Array, "3"]
    displacement: Float[Array, ""]


@jax.jit
def signed_distance(plane: PlaneData, p: Float[Array, "3"]) -> Float[Array, ""]:
    """Return the signed distance from the plane to the point. Positive means the
    point lies above the plane.
    """
    return jnp.dot(plane.normal, p) - plane.displacement


def intersect_line_with_plane(
    p_a: Float[Array, "3"], p_b: Float[Array, "3"], h: PlaneData
) -> Float[Array, "3"]:
    """Return the intersection point of a line segment with a plane."""
    a = signed_distance(h, p_a)
    b = signed_distance(h, p_b)
    wa = b / (b - a)
    wb = 1.0 - wa
    return wa * p_a + wb * p_b


def clip_polygon_by_halfspace(
    polygon: List[Float[Array, "3"]], h: PlaneData
) -> List[Float[Array, "3"]]:
    """Clip a polygon by a halfspace defined by a plane using the inner loop of the Sutherland-Hodgman algorithm."""

    def add_unique_vertex(
        output_polygon: List[Float[Array, "3"]], vertex: Float[Array, "3"]
    ) -> None:
        """Add a vertex to the polygon if it is not already present. This is done to avoid adding duplicate vertices due to numerical imprecision."""

        def near(a, b, eps=1e-14):
            return jnp.sum((a - b) ** 2) < eps**2

        if not output_polygon or not near(output_polygon[-1], vertex):
            output_polygon.append(vertex)

    output = []

    # Iterate over the edges of the polygon
    for i in range(len(polygon)):
        current_vertex = polygon[i]
        previous_vertex = polygon[i - 1]  # Automatically wraps to the last vertex

        current_vertex_in_halfspace = signed_distance(h, current_vertex) <= 0.0
        previous_vertex_in_halfspace = signed_distance(h, previous_vertex) <= 0.0

        # The edge is entirely inside the halfspace
        if current_vertex_in_halfspace and previous_vertex_in_halfspace:
            # Append only the current vertex (assuming the previous vertex is already added)
            add_unique_vertex(output, current_vertex)

        # The edge starts inside but crosses out of the half space
        elif previous_vertex_in_halfspace and not current_vertex_in_halfspace:
            # Append the only intersection point (assuming the previous vertex is already added)
            intersection = intersect_line_with_plane(current_vertex, previous_vertex, h)
            add_unique_vertex(output, intersection)

        # The edge starts outside but crosses into the half space
        elif not previous_vertex_in_halfspace and current_vertex_in_halfspace:
            # Append both the intersection point and the current vertex
            # (the previous vertex is outside so we don't add it)
            intersection = intersect_line_with_plane(current_vertex, previous_vertex, h)
            add_unique_vertex(output, intersection)
            add_unique_vertex(output, current_vertex)

        # The edge is entirely outside the halfspace
        else:
            # Append nothing
            pass

    return output


def clip_triangle_by_tetrahedron(
    triangle_mesh: TriangleMesh,
    tetrahedral_mesh: VolumeMesh,
    triangle_index: int | Int[Array, ""],
    tetrahedron_index: int | Int[Array, ""],
    triangle_frame_to_tetrahedron_frame: jaxlie.SE3,
) -> List[Float[Array, "3"]]:
    """Clip a triangle by a tetrahedron."""
    triangle = triangle_mesh.elements[triangle_index]
    # Initialize the intersection polygon with the triangle vertices in the tetrahedron frame
    polygon = [
        triangle_frame_to_tetrahedron_frame @ triangle_mesh.vertices[i]
        for i in triangle
    ]

    tetrahedron = tetrahedral_mesh.elements[tetrahedron_index]
    tetrahedron_vertices = tetrahedral_mesh.vertices[tetrahedron]

    faces = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]
    for face in faces:
        p_a, p_b, p_c = (tetrahedron_vertices[index] for index in face)
        face_normal = jnp.cross(p_b - p_a, p_c - p_a)
        face_normal /= jnp.linalg.norm(face_normal)
        halfspace = PlaneData(
            normal=face_normal, displacement=jnp.dot(face_normal, p_a)
        )
        polygon = clip_polygon_by_halfspace(polygon, halfspace)
        if len(polygon) < 3:
            return []

    return polygon


def compute_polygon_centroid(
    polygon: List[Float[Array, "3"]], normal: Float[Array, "3"]
) -> Float[Array, "3"]:
    n = len(polygon)

    if n <= 3:
        return sum(polygon) / n

    # Decompose the polygon into a fan of triangles around the first vertex
    total_weight = 0.0
    v0 = polygon[0]
    weighted_sum = jnp.zeros(3)

    for v1, v2 in zip(polygon[1:-1], polygon[2:]):
        weight = jnp.dot(jnp.cross(v1 - v0, v2 - v0), normal)
        weighted_sum += weight * (v0 + v1 + v2) / 3
        total_weight += weight

    # If the polygon is degenerate, fall back to returning the average of the vertices
    if abs(total_weight) < 1e-14:
        return sum(polygon) / n
    else:
        return weighted_sum / total_weight


def triangulate_polygon(
    polygon: List[Float[Array, "3"]],
    normal: Float[Array, "3"],
) -> Tuple[List[Int[Array, "3"]], List[Float[Array, "3"]]]:
    n = len(polygon)
    if n < 3:
        return [], []

    centroid = compute_polygon_centroid(polygon, normal)
    centroid_index = n

    vertices = polygon + [centroid]

    # Iterate over consecutive vertices (edges) in the polygon
    triangles = []
    for i in range(n):
        triangle = jnp.array([i, (i + 1) % n, centroid_index])
        triangles.append(triangle)

    return triangles, vertices


def sample_volume_field_on_surface(
    triangle_mesh: TriangleMesh,
    tetrahedral_mesh: VolumeMesh,
    triangle_mesh_pose: jaxlie.SE3,
    tetrahedral_mesh_pose: jaxlie.SE3,
) -> None:
    triangle_mesh_to_tetrahedral_mesh = (
        tetrahedral_mesh_pose.inverse() @ triangle_mesh_pose
    )

    polygons = []
    for tetrahedron_index in range(tetrahedral_mesh.num_elements()):
        for triangle_index in range(triangle_mesh.num_elements()):
            intersection_polygon = clip_triangle_by_tetrahedron(
                triangle_mesh,
                tetrahedral_mesh,
                triangle_index,
                tetrahedron_index,
                triangle_mesh_to_tetrahedral_mesh,
            )

            if len(intersection_polygon) >= 3:
                polygons.append(intersection_polygon)


def compute_contact_surface(
    triangle_mesh: TriangleMesh,
    tetrahedral_mesh: VolumeMesh,
):
    pass
