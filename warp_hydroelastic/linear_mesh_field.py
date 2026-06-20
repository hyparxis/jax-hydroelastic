from dataclasses import dataclass
from typing import Any

import warp as wp

from warp_hydroelastic.mesh import _as_warp_array
from warp_hydroelastic.volume_mesh import VolumeMesh, VolumeMeshData


@wp.struct
class LinearMeshFieldData:
    mesh: VolumeMeshData
    values: wp.array(dtype=float)
    gradients: wp.array(dtype=wp.vec3)
    constants: wp.array(dtype=float)


@wp.func
def _barycentric_gradient(
    mesh: VolumeMeshData,
    e: int,
    v: int,
) -> wp.vec3:
    element = mesh.elements[e]

    V = mesh.vertices[element[v]]
    A = mesh.vertices[element[(v + 1) % 4]]
    B = mesh.vertices[element[(v + 2) % 4]]
    C = mesh.vertices[element[(v + 3) % 4]]

    AV = V - A
    AB = B - A
    AC = C - A

    area_vector = wp.cross(AB, AC)
    signed_volume = wp.dot(area_vector, AV)

    return area_vector / signed_volume


@wp.kernel
def _compute_volume_field_affine_kernel(
    mesh: VolumeMeshData,
    values: wp.array(dtype=float),
    gradients: wp.array(dtype=wp.vec3),
    constants: wp.array(dtype=float),
):
    e = wp.tid()
    element = mesh.elements[e]

    gradient = wp.vec3(0.0, 0.0, 0.0)
    for v in range(4):
        gradient = gradient + values[element[v]] * _barycentric_gradient(mesh, e, v)

    gradients[e] = gradient
    constants[e] = values[element[0]] - wp.dot(gradient, mesh.vertices[element[0]])


@wp.kernel
def _compute_volume_field_constants_kernel(
    mesh: VolumeMeshData,
    values: wp.array(dtype=float),
    gradients: wp.array(dtype=wp.vec3),
    constants: wp.array(dtype=float),
):
    e = wp.tid()
    element = mesh.elements[e]
    constants[e] = values[element[0]] - wp.dot(gradients[e], mesh.vertices[element[0]])


@dataclass(frozen=True)
class LinearMeshField:
    mesh: VolumeMesh
    values: wp.array
    gradients: wp.array
    constants: wp.array

    @classmethod
    def create(
        cls,
        mesh: VolumeMesh,
        values: Any,
        gradients: Any | None = None,
    ):
        values = _as_warp_array(values, float, mesh.device)

        if gradients is None:
            gradients = wp.empty(mesh.num_elements(), dtype=wp.vec3, device=mesh.device)
            constants = wp.empty(mesh.num_elements(), dtype=float, device=mesh.device)
            wp.launch(
                _compute_volume_field_affine_kernel,
                dim=mesh.num_elements(),
                inputs=[mesh.data(), values, gradients, constants],
                device=mesh.device,
            )
        else:
            gradients = _as_warp_array(gradients, wp.vec3, mesh.device)
            constants = wp.empty(mesh.num_elements(), dtype=float, device=mesh.device)
            wp.launch(
                _compute_volume_field_constants_kernel,
                dim=mesh.num_elements(),
                inputs=[mesh.data(), values, gradients, constants],
                device=mesh.device,
            )

        return cls(
            mesh=mesh,
            values=values,
            gradients=gradients,
            constants=constants,
        )

    def value_at_vertex(self, vertex_index: int) -> float:
        return float(self.values.numpy()[vertex_index])

    def gradient_at_element(self, element_index: int):
        return self.gradients.numpy()[element_index]

    def value_at_cartesian_point(self, element_index: int, point: Any) -> float:
        gradient = self.gradients.numpy()[element_index]
        constant = self.constants.numpy()[element_index]
        return float(gradient.dot(point) + constant)

    def data(self) -> LinearMeshFieldData:
        data = LinearMeshFieldData()
        data.mesh = self.mesh.data()
        data.values = self.values
        data.gradients = self.gradients
        data.constants = self.constants
        return data
