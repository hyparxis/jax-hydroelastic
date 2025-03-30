from typing import List

import jax
import jax.numpy as jnp

from jax_hydroelastic.triangle_mesh import TriangleMesh
from jax_hydroelastic.volume_mesh import VolumeMesh


class Plane:
    """A plane defined by the implicit equation: `P(x⃗) = n̂⋅x⃗ - d = 0`"""

    def __init__(
        self, normal: jax.Array, point: jax.Array, is_normalized: bool = False
    ):
        assert normal.shape == (3,)
        assert point.shape == (3,)

        if not is_normalized:
            magnitude = jnp.linalg.norm(normal)
            assert magnitude > 1e-10

            normal = jax.lax.div(normal, magnitude)

        self.normal = normal
        self.displacement = jnp.dot(normal, point)

    def signed_distance(self, p: jax.Array) -> jax.Array:
        """Return the signed distance from the plane to the point. Positive means the
        point lies above the plane.
        """
        return jnp.dot(self.normal, p) - self.displacement


def intersect_line_with_plane(p_a: jax.Array, p_b: jax.Array, h: Plane) -> jax.Array:
    """Return the intersection point of a line segment with a plane."""
    a = h.signed_distance(p_a)
    b = h.signed_distance(p_b)
    wa = b / (b - a)
    wb = 1.0 - wa
    return wa * p_a + wb * p_b


def clip_polygon_by_halfspace(polygon: List[jax.Array], h: Plane) -> List[jax.Array]:
    """Clip a polygon by a halfspace defined by a plane using the inner loop of the Sutherland-Hodgman algorithm."""

    def add_unique_vertex(output_polygon: List[jax.Array], vertex: jax.Array) -> None:
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

        current_vertex_in_halfspace = h.signed_distance(current_vertex) <= 0.0
        previous_vertex_in_halfspace = h.signed_distance(previous_vertex) <= 0.0

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
    triangle_index: int,
    tetrahedron_index: int,
    rotation: jax.Array,
    translation: jax.Array,
) -> jax.Array:
    """Clip a triangle by a tetrahedron."""
    triangle = triangle_mesh.get_element(triangle_index)
    polygon = [
        rotation @ triangle_mesh.get_vertex(i) + translation for i in triangle.indices
    ]

    tetrahedron = tetrahedral_mesh.get_element(tetrahedron_index)
    tetrahedron_vertices = [tetrahedral_mesh.get_vertex(i) for i in tetrahedron.indices]

    faces = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]
    for face in faces:
        p_a, p_b, p_c = (tetrahedron_vertices[index] for index in face)
        face_normal = jnp.cross(p_b - p_a, p_c - p_a)
        halfspace = Plane(face_normal, p_a)
        polygon = clip_polygon_by_halfspace(polygon, halfspace)
        if len(polygon) < 3:
            return []

    return polygon
