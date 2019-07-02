from __future__ import annotations

import json
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

        def get_transform(node_json):
            if "matrix" in node_json:
                matrix_json = node_json["matrix"]
                matrix = tuple(
                    matrix_json[0:3]
                    + matrix_json[4:7]
                    + matrix_json[8:11]
                    + matrix_json[12:15]
                )
                if matrix == (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0):
                    return None
                return matrix

            # scale = tuple(node_json.get("scale", (1.0, 1.0, 1.0)))
            # rotation = tuple(node_json.get("rotation", (0.0, 0.0, 0.0, 1.0)))
            # translation = tuple(node_json.get("translation", (0.0, 0.0, 0.0)))
            raise RuntimeError("unimplemented")

        node_transforms = []
        node_index_to_flattened_node_index = []
        for node_json in gltf_json["nodes"]:
            transform = get_transform(node_json)
            node_index_to_flattened_node_index.append(len(node_transforms))
            if transform:
                node_transforms.append(transform)

        transform_stages = []
        mesh_index_to_flattened_node_indices = [[]] * len(gltf_json["meshes"])
        node_index_to_parent_index = [None] * len(gltf_json["nodes"])

        def visit(
            node_indices, parent_index: typing.Optional[int] = None, depth: int = 0
        ):
            for node_index in node_indices:
                node_index_to_parent_index[node_index] = parent_index
                current_depth = depth
                current_parent_index = None
                if node_index in node_index_to_flattened_node_index:
                    while len(transform_stages) <= current_depth:
                        transform_stages.append([])

                    # if parent_index:
                    transform_stages[current_depth].append((parent_index, node_index))

                    current_parent_index = node_index
                    current_depth += 1

                node_json = gltf_json["nodes"][node_index]

                if "mesh" in node_json:
                    mesh_index_to_flattened_node_indices[node_json["mesh"]].append(
                        node_index_to_flattened_node_index[parent_index]
                    )

                visit(
                    node_json.get("children", []),
                    parent_index=current_parent_index,
                    depth=current_depth,
                )

        visit(gltf_json["scenes"][0]["nodes"])
        print(mesh_index_to_flattene_node_indices)
