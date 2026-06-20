from dataclasses import dataclass
from typing import Any

import warp as wp

from warp_hydroelastic.mesh import Mesh, _as_warp_array, _resolve_device


@wp.struct
class VolumeMeshData:
    elements: wp.array(dtype=wp.vec4i)
    vertices: wp.array(dtype=wp.vec3)


@dataclass(frozen=True)
class VolumeMesh(Mesh):
    """Represents a volume mesh as a list of tetrahedra."""

    elements: wp.array
    vertices: wp.array
    device: str

    @classmethod
    def create(
        cls,
        tetrahedra: Any,
        vertices: Any,
        device: str | None = None,
    ):
        device = _resolve_device(device)
        return cls(
            elements=_as_warp_array(tetrahedra, wp.vec4i, device),
            vertices=_as_warp_array(vertices, wp.vec3, device),
            device=device,
        )

    @classmethod
    def num_vertices_per_element(cls) -> int:
        return 4

    def data(self) -> VolumeMeshData:
        data = VolumeMeshData()
        data.elements = self.elements
        data.vertices = self.vertices
        return data
