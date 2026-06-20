from dataclasses import dataclass
from typing import Any

import warp as wp

from warp_hydroelastic.mesh import Mesh, _as_warp_array, _resolve_device


@wp.struct
class TriangleMeshData:
    elements: wp.array(dtype=wp.vec3i)
    vertices: wp.array(dtype=wp.vec3)
    face_normals: wp.array(dtype=wp.vec3)


def _triangle_mesh_data(
    elements: wp.array,
    vertices: wp.array,
    face_normals: wp.array,
) -> TriangleMeshData:
    data = TriangleMeshData()
    data.elements = elements
    data.vertices = vertices
    data.face_normals = face_normals
    return data


@wp.kernel
def _compute_face_normals_kernel(
    mesh: TriangleMeshData,
    face_normals: wp.array(dtype=wp.vec3),
):
    e = wp.tid()
    triangle = mesh.elements[e]

    v0 = mesh.vertices[triangle[0]]
    v1 = mesh.vertices[triangle[1]]
    v2 = mesh.vertices[triangle[2]]

    face_normal = wp.cross(v1 - v0, v2 - v0)
    norm = wp.length(face_normal)

    if norm > 1.0e-14:
        face_normals[e] = face_normal / norm
    else:
        face_normals[e] = wp.vec3(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TriangleMesh(Mesh):
    """Represents a triangle mesh as a list of triangles."""

    elements: wp.array
    vertices: wp.array
    face_normals: wp.array
    device: str

    @classmethod
    def create(
        cls,
        triangles: Any,
        vertices: Any,
        face_normals: Any | None = None,
        device: str | None = None,
    ):
        device = _resolve_device(device)
        elements = _as_warp_array(triangles, wp.vec3i, device)
        vertices = _as_warp_array(vertices, wp.vec3, device)

        if face_normals is None:
            face_normals = wp.empty(elements.shape[0], dtype=wp.vec3, device=device)
            wp.launch(
                _compute_face_normals_kernel,
                dim=elements.shape[0],
                inputs=[
                    _triangle_mesh_data(elements, vertices, face_normals),
                    face_normals,
                ],
                device=device,
            )
        else:
            face_normals = _as_warp_array(face_normals, wp.vec3, device)

        return cls(
            elements=elements,
            vertices=vertices,
            face_normals=face_normals,
            device=device,
        )

    @classmethod
    def num_vertices_per_element(cls) -> int:
        return 3

    def face_normal(self, index: int) -> wp.vec3:
        normal = self.face_normals.numpy()[index]
        return wp.vec3(float(normal[0]), float(normal[1]), float(normal[2]))

    def data(self) -> TriangleMeshData:
        return _triangle_mesh_data(self.elements, self.vertices, self.face_normals)
