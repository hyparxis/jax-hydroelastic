from typing import Tuple

import jax
import jax.numpy as jnp


def transform_point(
    p: jax.Array, rotation: jax.Array, translation: jax.Array
) -> jax.Array:
    return jax.lax.batch_matmul(p, rotation) + translation


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


def clip_polygon_by_halfspace(polygon: jax.Array, h: Plane) -> Tuple[jax.Array, int]:
    """Clip a polygon by a halfspace defined by a plane."""

    # This is the inner loop of a modified Sutherland-Hodgman algorithm for clipping a
    # polygon.

    input_size = polygon.shape[0]
    output_size = 0

    # Always return a polygon with 7 vertices so this function can be jit compiled.
    output = jnp.nan * jnp.ones((7, 3))

    for i in range(input_size):
        current_point = polygon[i]
        previous_point = polygon[(i - 1 + input_size) % input_size]

        current_point_contained = h.signed_distance(current_point) <= 0.0
        previous_point_contained = h.signed_distance(previous_point) <= 0.0

        if current_point_contained:
            if not previous_point_contained:
                intersection = intersect_line_with_plane(
                    current_point, previous_point, h
                )
                output = output.at[output_size, :].set(intersection)
                output_size += 1
            output = output.at[output_size, :].set(current_point)
            output_size += 1
        elif previous_point_contained:
            intersection = intersect_line_with_plane(current_point, previous_point, h)
            output = output.at[output_size, :].set(intersection)
            output_size += 1

    return output, output_size


def clip_triangle_by_tetrahedron(
    triangle: jax.Array,
    tetrahedron: jax.Array,
    rotation: jax.Array,
    translation: jax.Array,
) -> Tuple[jax.Array, int]:
    """Clip a triangle by a tetrahedron."""
    output = jnp.nan * jnp.ones((7, 3))
    output_size = 3
