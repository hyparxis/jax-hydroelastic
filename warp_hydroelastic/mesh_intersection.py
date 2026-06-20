from dataclasses import dataclass

import numpy as np
import warp as wp

from warp_hydroelastic.linear_mesh_field import LinearMeshField, LinearMeshFieldData
from warp_hydroelastic.triangle_mesh import TriangleMesh, TriangleMeshData
from warp_hydroelastic.volume_mesh import VolumeMesh, VolumeMeshData

MAX_POLYGON_VERTICES = 7
MAX_CONTACT_VERTICES = MAX_POLYGON_VERTICES + 1
MAX_CONTACT_TRIANGLES = MAX_POLYGON_VERTICES


@wp.struct
class PlaneData:
    """Data for a plane defined by the implicit equation: `P(x) = n dot x - d = 0`"""

    normal: wp.vec3
    displacement: float


@wp.struct
class FixedPolygonData:
    vertices: wp.array(dtype=wp.vec3)
    size: wp.array(dtype=int)


@wp.struct
class SampledPolygonData:
    triangles: wp.array(dtype=wp.vec3i)
    vertices: wp.array(dtype=wp.vec3)
    pressures: wp.array(dtype=float)
    size: wp.array(dtype=int)


@wp.struct
class ContactSurfaceData:
    polygon_vertices: wp.array3d(dtype=wp.vec3)
    polygon_sizes: wp.array2d(dtype=int)
    surface_triangles: wp.array3d(dtype=wp.vec3i)
    surface_vertices: wp.array3d(dtype=wp.vec3)
    surface_pressures: wp.array3d(dtype=float)
    surface_sizes: wp.array2d(dtype=int)


@dataclass(frozen=True)
class FixedPolygon:
    vertices: wp.array
    size: wp.array

    def num_vertices(self) -> int:
        return int(self.size.numpy()[0])

    def numpy(self) -> np.ndarray:
        return self.vertices.numpy()[: self.num_vertices()]

    def data(self) -> FixedPolygonData:
        data = FixedPolygonData()
        data.vertices = self.vertices
        data.size = self.size
        return data


@dataclass(frozen=True)
class SampledPolygon:
    triangles: wp.array
    vertices: wp.array
    pressures: wp.array
    size: wp.array

    def num_triangles(self) -> int:
        return int(self.size.numpy()[0])

    def numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.num_triangles()
        return (
            self.triangles.numpy()[:n],
            self.vertices.numpy()[: n + 1],
            self.pressures.numpy()[: n + 1],
        )

    def data(self) -> SampledPolygonData:
        data = SampledPolygonData()
        data.triangles = self.triangles
        data.vertices = self.vertices
        data.pressures = self.pressures
        data.size = self.size
        return data


@dataclass(frozen=True)
class ContactSurface:
    polygon_vertices: wp.array
    polygon_sizes: wp.array
    surface_triangles: wp.array
    surface_vertices: wp.array
    surface_pressures: wp.array
    surface_sizes: wp.array

    def data(self) -> ContactSurfaceData:
        data = ContactSurfaceData()
        data.polygon_vertices = self.polygon_vertices
        data.polygon_sizes = self.polygon_sizes
        data.surface_triangles = self.surface_triangles
        data.surface_vertices = self.surface_vertices
        data.surface_pressures = self.surface_pressures
        data.surface_sizes = self.surface_sizes
        return data


@wp.func
def signed_distance(plane: PlaneData, p: wp.vec3) -> float:
    """Return the signed distance from the plane to the point."""
    return wp.dot(plane.normal, p) - plane.displacement


@wp.func
def intersect_line_with_plane(p_a: wp.vec3, p_b: wp.vec3, h: PlaneData) -> wp.vec3:
    """Return the intersection point of a line segment with a plane."""
    a = signed_distance(h, p_a)
    b = signed_distance(h, p_b)
    wa = b / (b - a)
    wb = 1.0 - wa
    return wa * p_a + wb * p_b


@wp.func
def _near(a: wp.vec3, b: wp.vec3) -> bool:
    delta = a - b
    return wp.dot(delta, delta) < 1.0e-28


@wp.func
def _add_unique_vertex(
    output_polygon: wp.array(dtype=wp.vec3), count: int, vertex: wp.vec3
) -> int:
    """Add a vertex to the polygon if it is not already present."""
    should_add = False

    if count == 0:
        should_add = True
    else:
        if not _near(output_polygon[count - 1], vertex):
            should_add = True

    if should_add:
        if count < MAX_POLYGON_VERTICES:
            output_polygon[count] = vertex
            count = count + 1

    return count


@wp.func
def _clip_polygon_by_halfspace(
    polygon: wp.array(dtype=wp.vec3),
    polygon_size: int,
    h: PlaneData,
    output: wp.array(dtype=wp.vec3),
) -> int:
    """Clip a polygon by a halfspace using the Sutherland-Hodgman inner loop."""
    output_size = int(0)

    for i in range(MAX_POLYGON_VERTICES):
        if i < polygon_size:
            current_vertex = polygon[i]

            previous_index = polygon_size - 1
            if i > 0:
                previous_index = i - 1
            previous_vertex = polygon[previous_index]

            current_vertex_in_halfspace = signed_distance(h, current_vertex) <= 0.0
            previous_vertex_in_halfspace = (
                signed_distance(h, previous_vertex) <= 0.0
            )

            if current_vertex_in_halfspace:
                if previous_vertex_in_halfspace:
                    output_size = _add_unique_vertex(
                        output, output_size, current_vertex
                    )
                else:
                    intersection = intersect_line_with_plane(
                        current_vertex, previous_vertex, h
                    )
                    output_size = _add_unique_vertex(output, output_size, intersection)
                    output_size = _add_unique_vertex(
                        output, output_size, current_vertex
                    )
            else:
                if previous_vertex_in_halfspace:
                    intersection = intersect_line_with_plane(
                        current_vertex, previous_vertex, h
                    )
                    output_size = _add_unique_vertex(output, output_size, intersection)

    return output_size


@wp.func
def _tetrahedron_face_plane(
    tetrahedron_vertices: wp.array(dtype=wp.vec3), face_index: int
) -> PlaneData:
    ia = int(1)
    ib = int(2)
    ic = int(3)

    if face_index == 1:
        ia = 0
        ib = 3
        ic = 2
    if face_index == 2:
        ia = 0
        ib = 1
        ic = 3
    if face_index == 3:
        ia = 0
        ib = 2
        ic = 1

    p_a = tetrahedron_vertices[ia]
    p_b = tetrahedron_vertices[ib]
    p_c = tetrahedron_vertices[ic]

    face_normal = wp.cross(p_b - p_a, p_c - p_a)
    norm = wp.length(face_normal)
    if norm > 1.0e-14:
        face_normal = face_normal / norm
    else:
        face_normal = wp.vec3(0.0, 0.0, 0.0)

    halfspace = PlaneData()
    halfspace.normal = face_normal
    halfspace.displacement = wp.dot(face_normal, p_a)
    return halfspace


@wp.func
def _clip_triangle_by_tetrahedron(
    triangle_mesh: TriangleMeshData,
    tetrahedral_mesh: VolumeMeshData,
    triangle_index: int,
    tetrahedron_index: int,
    triangle_frame_to_tetrahedron_frame: wp.transform,
    polygon: wp.array(dtype=wp.vec3),
) -> int:
    """Clip a triangle by a tetrahedron into fixed-capacity polygon storage."""
    triangle = triangle_mesh.elements[triangle_index]

    polygon[0] = wp.transform_point(
        triangle_frame_to_tetrahedron_frame, triangle_mesh.vertices[triangle[0]]
    )
    polygon[1] = wp.transform_point(
        triangle_frame_to_tetrahedron_frame, triangle_mesh.vertices[triangle[1]]
    )
    polygon[2] = wp.transform_point(
        triangle_frame_to_tetrahedron_frame, triangle_mesh.vertices[triangle[2]]
    )
    polygon_size = int(3)

    tetrahedron = tetrahedral_mesh.elements[tetrahedron_index]
    tetrahedron_local_vertices = wp.zeros(shape=4, dtype=wp.vec3)
    for i in range(4):
        tetrahedron_local_vertices[i] = tetrahedral_mesh.vertices[tetrahedron[i]]

    scratch = wp.zeros(shape=MAX_POLYGON_VERTICES, dtype=wp.vec3)

    for face_index in range(4):
        if polygon_size >= 3:
            halfspace = _tetrahedron_face_plane(tetrahedron_local_vertices, face_index)
            polygon_size = _clip_polygon_by_halfspace(
                polygon, polygon_size, halfspace, scratch
            )

            for i in range(MAX_POLYGON_VERTICES):
                polygon[i] = scratch[i]
        else:
            polygon_size = 0

    return polygon_size


@wp.func
def _compute_polygon_centroid(
    polygon: wp.array(dtype=wp.vec3), polygon_size: int, normal: wp.vec3
) -> wp.vec3:
    centroid = wp.vec3(0.0, 0.0, 0.0)

    if polygon_size <= 3:
        point_sum = wp.vec3(0.0, 0.0, 0.0)
        for i in range(MAX_POLYGON_VERTICES):
            if i < polygon_size:
                point_sum = point_sum + polygon[i]
        centroid = point_sum / float(polygon_size)
    else:
        total_weight = float(0.0)
        weighted_sum = wp.vec3(0.0, 0.0, 0.0)
        v0 = polygon[0]

        for i in range(MAX_POLYGON_VERTICES - 2):
            if i < polygon_size - 2:
                v1 = polygon[i + 1]
                v2 = polygon[i + 2]
                weight = wp.dot(wp.cross(v1 - v0, v2 - v0), normal)
                weighted_sum = weighted_sum + weight * (v0 + v1 + v2) / 3.0
                total_weight = total_weight + weight

        if wp.abs(total_weight) < 1.0e-14:
            point_sum = wp.vec3(0.0, 0.0, 0.0)
            for i in range(MAX_POLYGON_VERTICES):
                if i < polygon_size:
                    point_sum = point_sum + polygon[i]
            centroid = point_sum / float(polygon_size)
        else:
            centroid = weighted_sum / total_weight

    return centroid


@wp.func
def _sample_pressure_field_on_polygon(
    polygon: wp.array(dtype=wp.vec3),
    polygon_size: int,
    field: LinearMeshFieldData,
    tetrahedron_index: int,
    normal: wp.vec3,
    triangles: wp.array(dtype=wp.vec3i),
    vertices: wp.array(dtype=wp.vec3),
    pressures: wp.array(dtype=float),
) -> int:
    surface_size = int(0)

    if polygon_size < 3:
        surface_size = 0
    else:
        centroid = _compute_polygon_centroid(polygon, polygon_size, normal)
        gradient = field.gradients[tetrahedron_index]
        constant = field.constants[tetrahedron_index]

        for i in range(MAX_CONTACT_VERTICES):
            if i < polygon_size:
                vertices[i] = polygon[i]
                pressures[i] = wp.dot(gradient, polygon[i]) + constant
            if i == polygon_size:
                vertices[i] = centroid
                pressures[i] = wp.dot(gradient, centroid) + constant

        for i in range(MAX_CONTACT_TRIANGLES):
            if i < polygon_size:
                triangles[i] = wp.vec3i(i, (i + 1) % polygon_size, polygon_size)

        surface_size = polygon_size

    return surface_size


@wp.kernel
def _clip_triangle_by_tetrahedron_kernel(
    triangle_mesh: TriangleMeshData,
    tetrahedral_mesh: VolumeMeshData,
    triangle_index: int,
    tetrahedron_index: int,
    triangle_frame_to_tetrahedron_frame: wp.transform,
    output_polygon: FixedPolygonData,
):
    polygon = wp.zeros(shape=MAX_POLYGON_VERTICES, dtype=wp.vec3)
    size = _clip_triangle_by_tetrahedron(
        triangle_mesh,
        tetrahedral_mesh,
        triangle_index,
        tetrahedron_index,
        triangle_frame_to_tetrahedron_frame,
        polygon,
    )

    output_polygon.size[0] = size
    for i in range(MAX_POLYGON_VERTICES):
        output_polygon.vertices[i] = polygon[i]


@wp.kernel
def _sample_pressure_field_on_polygon_kernel(
    polygon_data: FixedPolygonData,
    field: LinearMeshFieldData,
    tetrahedron_index: int,
    normal: wp.vec3,
    sampled_polygon: SampledPolygonData,
):
    polygon = wp.zeros(shape=MAX_POLYGON_VERTICES, dtype=wp.vec3)
    for i in range(MAX_POLYGON_VERTICES):
        polygon[i] = polygon_data.vertices[i]

    triangles = wp.zeros(shape=MAX_CONTACT_TRIANGLES, dtype=wp.vec3i)
    vertices = wp.zeros(shape=MAX_CONTACT_VERTICES, dtype=wp.vec3)
    pressures = wp.zeros(shape=MAX_CONTACT_VERTICES, dtype=float)

    size = _sample_pressure_field_on_polygon(
        polygon,
        polygon_data.size[0],
        field,
        tetrahedron_index,
        normal,
        triangles,
        vertices,
        pressures,
    )

    sampled_polygon.size[0] = size
    for i in range(MAX_CONTACT_TRIANGLES):
        sampled_polygon.triangles[i] = triangles[i]
    for i in range(MAX_CONTACT_VERTICES):
        sampled_polygon.vertices[i] = vertices[i]
        sampled_polygon.pressures[i] = pressures[i]


@wp.kernel
def _sample_volume_field_on_surface_kernel(
    triangle_mesh: TriangleMeshData,
    field: LinearMeshFieldData,
    triangle_frame_to_tetrahedron_frame: wp.transform,
    contact_surface: ContactSurfaceData,
):
    tetrahedron_index, triangle_index = wp.tid()

    polygon = wp.zeros(shape=MAX_POLYGON_VERTICES, dtype=wp.vec3)
    polygon_size = _clip_triangle_by_tetrahedron(
        triangle_mesh,
        field.mesh,
        triangle_index,
        tetrahedron_index,
        triangle_frame_to_tetrahedron_frame,
        polygon,
    )

    contact_surface.polygon_sizes[tetrahedron_index, triangle_index] = polygon_size
    for i in range(MAX_POLYGON_VERTICES):
        contact_surface.polygon_vertices[tetrahedron_index, triangle_index, i] = (
            polygon[i]
        )

    surface_size = int(0)
    if polygon_size >= 3:
        normal = wp.transform_vector(
            triangle_frame_to_tetrahedron_frame,
            triangle_mesh.face_normals[triangle_index],
        )
        norm = wp.length(normal)
        if norm > 1.0e-14:
            normal = normal / norm

        triangles = wp.zeros(shape=MAX_CONTACT_TRIANGLES, dtype=wp.vec3i)
        vertices = wp.zeros(shape=MAX_CONTACT_VERTICES, dtype=wp.vec3)
        pressures = wp.zeros(shape=MAX_CONTACT_VERTICES, dtype=float)

        surface_size = _sample_pressure_field_on_polygon(
            polygon,
            polygon_size,
            field,
            tetrahedron_index,
            normal,
            triangles,
            vertices,
            pressures,
        )

        for i in range(MAX_CONTACT_TRIANGLES):
            contact_surface.surface_triangles[tetrahedron_index, triangle_index, i] = (
                triangles[i]
            )
        for i in range(MAX_CONTACT_VERTICES):
            contact_surface.surface_vertices[tetrahedron_index, triangle_index, i] = (
                vertices[i]
            )
            contact_surface.surface_pressures[tetrahedron_index, triangle_index, i] = (
                pressures[i]
            )

    contact_surface.surface_sizes[tetrahedron_index, triangle_index] = surface_size


def _identity_transform() -> wp.transform:
    return wp.transform_identity()


def clip_triangle_by_tetrahedron(
    triangle_mesh: TriangleMesh,
    tetrahedral_mesh: VolumeMesh,
    triangle_index: int,
    tetrahedron_index: int,
    triangle_frame_to_tetrahedron_frame: wp.transform | None = None,
) -> FixedPolygon:
    """Clip a triangle by a tetrahedron."""
    if triangle_mesh.device != tetrahedral_mesh.device:
        raise ValueError("triangle_mesh and tetrahedral_mesh must be on the same device")

    if triangle_frame_to_tetrahedron_frame is None:
        triangle_frame_to_tetrahedron_frame = _identity_transform()

    polygon_vertices = wp.zeros(
        MAX_POLYGON_VERTICES, dtype=wp.vec3, device=triangle_mesh.device
    )
    polygon_size = wp.zeros(1, dtype=int, device=triangle_mesh.device)
    polygon = FixedPolygon(vertices=polygon_vertices, size=polygon_size)

    wp.launch(
        _clip_triangle_by_tetrahedron_kernel,
        dim=1,
        inputs=[
            triangle_mesh.data(),
            tetrahedral_mesh.data(),
            triangle_index,
            tetrahedron_index,
            triangle_frame_to_tetrahedron_frame,
            polygon.data(),
        ],
        device=triangle_mesh.device,
    )

    return polygon


def sample_pressure_field_on_polygon(
    polygon: FixedPolygon,
    field: LinearMeshField,
    tetrahedron_index: int,
    normal: wp.vec3,
) -> SampledPolygon:
    triangles = wp.zeros(
        MAX_CONTACT_TRIANGLES, dtype=wp.vec3i, device=field.mesh.device
    )
    vertices = wp.zeros(MAX_CONTACT_VERTICES, dtype=wp.vec3, device=field.mesh.device)
    pressures = wp.zeros(MAX_CONTACT_VERTICES, dtype=float, device=field.mesh.device)
    size = wp.zeros(1, dtype=int, device=field.mesh.device)
    sampled_polygon = SampledPolygon(
        triangles=triangles,
        vertices=vertices,
        pressures=pressures,
        size=size,
    )

    wp.launch(
        _sample_pressure_field_on_polygon_kernel,
        dim=1,
        inputs=[
            polygon.data(),
            field.data(),
            tetrahedron_index,
            normal,
            sampled_polygon.data(),
        ],
        device=field.mesh.device,
    )

    return sampled_polygon


def sample_volume_field_on_surface(
    triangle_mesh: TriangleMesh,
    field: LinearMeshField,
    triangle_mesh_pose: wp.transform | None = None,
    tetrahedral_mesh_pose: wp.transform | None = None,
) -> ContactSurface:
    if triangle_mesh.device != field.mesh.device:
        raise ValueError("triangle_mesh and field.mesh must be on the same device")

    if triangle_mesh_pose is None:
        triangle_mesh_pose = _identity_transform()
    if tetrahedral_mesh_pose is None:
        tetrahedral_mesh_pose = _identity_transform()

    triangle_mesh_to_tetrahedral_mesh = wp.transform_multiply(
        wp.transform_inverse(tetrahedral_mesh_pose), triangle_mesh_pose
    )

    shape = (field.mesh.num_elements(), triangle_mesh.num_elements())

    polygon_vertices = wp.zeros(
        shape=shape + (MAX_POLYGON_VERTICES,),
        dtype=wp.vec3,
        device=field.mesh.device,
    )
    polygon_sizes = wp.zeros(shape=shape, dtype=int, device=field.mesh.device)
    surface_triangles = wp.zeros(
        shape=shape + (MAX_CONTACT_TRIANGLES,),
        dtype=wp.vec3i,
        device=field.mesh.device,
    )
    surface_vertices = wp.zeros(
        shape=shape + (MAX_CONTACT_VERTICES,),
        dtype=wp.vec3,
        device=field.mesh.device,
    )
    surface_pressures = wp.zeros(
        shape=shape + (MAX_CONTACT_VERTICES,),
        dtype=float,
        device=field.mesh.device,
    )
    surface_sizes = wp.zeros(shape=shape, dtype=int, device=field.mesh.device)
    contact_surface = ContactSurface(
        polygon_vertices=polygon_vertices,
        polygon_sizes=polygon_sizes,
        surface_triangles=surface_triangles,
        surface_vertices=surface_vertices,
        surface_pressures=surface_pressures,
        surface_sizes=surface_sizes,
    )

    wp.launch(
        _sample_volume_field_on_surface_kernel,
        dim=shape,
        inputs=[
            triangle_mesh.data(),
            field.data(),
            triangle_mesh_to_tetrahedral_mesh,
            contact_surface.data(),
        ],
        device=field.mesh.device,
    )

    return contact_surface


def compute_contact_surface(
    triangle_mesh: TriangleMesh,
    tetrahedral_mesh: VolumeMesh,
):
    pass
