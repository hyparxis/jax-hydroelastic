from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    if wp.is_cuda_available():
        return "cuda:0"
    return "cpu"


def _as_numpy(data: Any, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(data, dtype=dtype))


def _as_warp_array(data: Any, dtype: type, device: str) -> wp.array:
    if isinstance(data, wp.array):
        return data

    if dtype in (float, wp.float32, wp.vec3):
        numpy_dtype = np.float32
    else:
        numpy_dtype = np.int32

    return wp.from_numpy(_as_numpy(data, numpy_dtype), dtype=dtype, device=device)


@dataclass(frozen=True)
class Mesh(ABC):
    elements: wp.array
    vertices: wp.array
    device: str

    @classmethod
    @abstractmethod
    def num_vertices_per_element(cls) -> int:
        """Return the number of vertices per element."""
        pass

    def num_elements(self) -> int:
        return self.elements.shape[0]

    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    def print(self) -> None:
        print("elements: ")
        print(self.elements.numpy())
        print("vertices: ")
        print(self.vertices.numpy())
