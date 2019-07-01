import json

from __future__ import annotations
import dataclasses
import typing


@dataclasses.dataclass(frozen=True)
class Bounds:
    min_extents: typing.Tuple[float, float, float]
    max_extents: typing.Tuple[float, float, float]


AffineMatrix = typing.Tuple[
    float, float, float, float, float, float, float, float, float, float, float, float
]


@dataclasses.dataclass(frozen=True)
class ScaleRotationTranslation:
    scale: typing.Tuple[float, float, float] = dataclasses.field(
        default=(1.0, 1.0, 1.0)
    )
    rotation: typing.Tuple[float, float, float, float] = dataclasses.field(
        default=(0.0, 0.0, 0.0, 1.0)
    )
    translation: typing.Tuple[float, float, float] = dataclasses.field(
        default=(0.0, 0.0, 0.0)
    )


@dataclasses.dataclass(eq=True, frozen=True, unsafe_hash=True)
class Node:
    transform: typing.Union[ScaleRotationTranslation, AffineMatrix]
    bounds: typing.Optional[Bounds] = dataclasses.field(default=None)
    children: typing.Collection["Node"] = dataclasses.field(default_factory=list)


class GltfModel:
    def __init__(self, file):
        gltf_json = json.load(file)
        scenes_json = gltf_json["scenes"]

        gltf_node_index_to_flat_node_index = []
        nodes = []

        def flatten_node_json(node_json):

            if "matrix" in node_json:
                pass
            elif (
                "scale" in node_json
                or "rotation" in node_json
                or "translation" in node_json
            ):
                pass
            else:
                continue

            # nodes.append(Node(

