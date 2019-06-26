import array
from typing import Optional
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mesh:
    indices: array.array
    positions: array.array
    texcoords: Optional[array.array] = field(default=None)
    normals: Optional[array.array] = field(default=None)
