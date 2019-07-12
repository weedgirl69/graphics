from __future__ import annotations
import base64
import collections
import json
import dataclasses
import typing

AffineTransform = typing.Tuple[
    float, float, float, float, float, float, float, float, float, float, float, float
]


@dataclasses.dataclass(frozen=True)
class GltfPrimitive:
    indices_accessor_index: typing.Optional[int]
    positions_accessor_index: int


@dataclasses.dataclass(frozen=True)
class TransformSequence:
    node_index_to_flattened_index: typing.Dict[int, int]
    transform_source_index_to_destination_index: typing.List[typing.Tuple[int, int]]


@dataclasses.dataclass(frozen=True)
class GltfScene:
    node_index_to_parent_index: typing.Dict[int, typing.Optional[int]]
    mesh_index_to_node_indices: typing.Dict[int, typing.List[int]]
    transform_sequence: TransformSequence
    mesh_offsets: typing.List[int]


def _get_node_transforms(gltf_json: typing.Dict) -> typing.List[AffineTransform]:
    def get_transform(node_json: typing.Dict) -> AffineTransform:
        if "matrix" in node_json:
            matrix_json = [float(component) for component in node_json["matrix"]]
            return (
                matrix_json[0],
                matrix_json[1],
                matrix_json[2],
                matrix_json[4],
                matrix_json[5],
                matrix_json[6],
                matrix_json[8],
                matrix_json[9],
                matrix_json[10],
                matrix_json[12],
                matrix_json[13],
                matrix_json[14],
            )

        scale_x, scale_y, scale_z = node_json.get("scale", (1.0, 1.0, 1.0))
        rotation_x, rotation_y, rotation_z, rotation_w = node_json.get(
            "rotation", (0.0, 0.0, 0.0, 1.0)
        )
        translation_x, translation_y, translation_z = node_json.get(
            "translation", (0.0, 0.0, 0.0)
        )

        # pylint: disable = invalid-name
        xx = rotation_x * rotation_x
        xy = rotation_x * rotation_y
        xz = rotation_x * rotation_z
        xw = rotation_x * rotation_w

        yy = rotation_y * rotation_y
        yz = rotation_y * rotation_z
        yw = rotation_y * rotation_w

        zz = rotation_z * rotation_z
        zw = rotation_z * rotation_w

        m00 = 1 - 2 * (yy + zz)
        m01 = 2 * (xy - zw)
        m02 = 2 * (xz + yw)

        m10 = 2 * (xy + zw)
        m11 = 1 - 2 * (xx + zz)
        m12 = 2 * (yz - xw)

        m20 = 2 * (xz - yw)
        m21 = 2 * (yz + xw)
        m22 = 1 - 2 * (xx + yy)

        return (
            scale_x * m00,
            scale_y * m01,
            scale_z * m02,
            translation_x,
            scale_x * m10,
            scale_y * m11,
            scale_z * m12,
            translation_y,
            scale_x * m20,
            scale_y * m21,
            scale_z * m22,
            translation_z,
        )

    return [get_transform(node_json) for node_json in gltf_json["nodes"]]


def _get_node_index_to_parent_index(
    *,
    nodes_json: typing.List[typing.Dict],
    node_indices: typing.List[int],
    parent_index: typing.Optional[int] = None,
    node_index_to_parent_index: typing.Dict[int, typing.Optional[int]] = None,
) -> typing.Dict[int, typing.Optional[int]]:
    if node_index_to_parent_index is None:
        node_index_to_parent_index = {}

    for node_index in node_indices:
        node_index_to_parent_index[node_index] = parent_index
        _get_node_index_to_parent_index(
            nodes_json=nodes_json,
            node_indices=nodes_json[node_index].get("children", []),
            parent_index=node_index,
            node_index_to_parent_index=node_index_to_parent_index,
        )

    return node_index_to_parent_index


def _get_mesh_index_to_node_indices(
    *,
    nodes_json: typing.List[typing.Dict],
    node_indices: typing.List[int],
    mesh_index_to_node_indices: typing.Dict[int, typing.List[int]] = None,
) -> typing.Dict[int, typing.List[int]]:
    if mesh_index_to_node_indices is None:
        mesh_index_to_node_indices = collections.defaultdict(list)

    for node_index in node_indices:
        node_json = nodes_json[node_index]

        if "mesh" in node_json:
            mesh_index_to_node_indices[node_json["mesh"]].append(node_index)

        _get_mesh_index_to_node_indices(
            nodes_json=nodes_json,
            node_indices=node_json.get("children", []),
            mesh_index_to_node_indices=mesh_index_to_node_indices,
        )

    return mesh_index_to_node_indices


def _get_transform_sequence(
    node_index_to_parent_index: typing.Dict[int, typing.Optional[int]],
    mesh_index_to_node_indices: typing.Dict[int, typing.List[int]],
) -> TransformSequence:
    node_index_to_flattened_index: typing.Dict[int, int] = {}
    transform_source_index_to_destination_index = []
    current_instance_index = 0

    def visit(node_index: int) -> None:
        parent_index = node_index_to_parent_index[node_index]
        if parent_index is not None:
            if parent_index not in node_index_to_flattened_index:
                node_index_to_flattened_index[parent_index] = len(
                    node_index_to_flattened_index
                )
                visit(parent_index)

            transform_source_index_to_destination_index.append(
                (
                    node_index_to_flattened_index[parent_index],
                    node_index_to_flattened_index[node_index],
                )
            )

    for node_indices in mesh_index_to_node_indices.values():
        for node_index in node_indices:
            node_index_to_flattened_index[node_index] = current_instance_index
            current_instance_index += 1

    for node_indices in mesh_index_to_node_indices.values():
        for node_index in node_indices:
            visit(node_index)

    return TransformSequence(
        node_index_to_flattened_index=node_index_to_flattened_index,
        transform_source_index_to_destination_index=transform_source_index_to_destination_index,
    )


def _get_accessors(
    *,
    accessors_json: typing.Dict,
    buffers_json: typing.Dict,
    buffer_views_json: typing.Dict,
    uri_resolver: typing.Callable[[str], bytes],
) -> typing.List[bytes]:
    buffers_data = []
    for buffer_json in buffers_json:
        uri: str = buffer_json["uri"]
        if uri.startswith("data:application/"):
            buffers_data.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            buffers_data.append(uri_resolver(uri))

    accessor_data = []
    for accessor_json in accessors_json:
        count = accessor_json["count"]
        byte_offset = accessor_json.get("byteOffset", 0)
        if "bufferView" in accessor_json:
            buffer_view_json = buffer_views_json[accessor_json["bufferView"]]
            byte_offset += buffer_view_json.get("byteOffset", 0)
            byte_count = buffer_view_json["byteLength"]

            component_size = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}[
                accessor_json["componentType"]
            ]
            component_count = {
                "SCALAR": 1,
                "VEC2": 2,
                "VEC3": 3,
                "VEC4": 4,
                "MAT2": 4,
                "MAT3": 9,
                "MAT4": 16,
            }[accessor_json["type"]]
            natural_stride = component_size * component_count
            byte_stride = buffer_view_json.get("byteStride", natural_stride)

            buffer_bytes = buffers_data[buffer_view_json["buffer"]]
            if byte_stride == natural_stride:
                accessor_data.append(
                    buffer_bytes[byte_offset : byte_offset + byte_count]
                )
            else:
                accessor_bytes = bytearray(byte_count)
                for i in range(count):
                    accessor_bytes[
                        i * natural_stride : (i + 1) * natural_stride
                    ] = buffer_bytes[i * byte_stride : i * byte_stride + natural_stride]
                accessor_data.append(accessor_bytes)
        else:
            accessor_data.append(bytes(count))

    return accessor_data


def _get_mesh_index_to_primitives(
    *, accessors: typing.List[bytes], meshes_json: typing.List[typing.Dict]
) -> typing.List[typing.list[GltfPrimitive]]:

    index_bytes = bytearray()

    for

    return [
        [
            GltfPrimitive(
                indices_accessor_index=primitive_json.get("indices", None),
                positions_accessor_index=primitive_json["attributes"]["POSITION"],
            )
            for primitive_json in mesh_json["primitives"]
        ]
        for mesh_json in meshes_json
    ]


class GltfModel:
    def __init__(self, file: typing.TextIO, uri_resolver):
        gltf_json = json.load(file)

        self.node_transforms = _get_node_transforms(gltf_json)
        self.scenes: typing.List[GltfScene] = []

        self.accessors = _get_accessors(
            accessors_json=gltf_json["accessors"],
            buffers_json=gltf_json["buffers"],
            buffer_views_json=gltf_json["bufferViews"],
            uri_resolver=uri_resolver,
        )

        self.meshes = _get_mesh_index_to_primitives(
            accessors=self.accessors, meshes_json=gltf_json["meshes"]
        )

        nodes_json = gltf_json["nodes"]
        for scene_json in gltf_json["scenes"]:
            scene_node_indices = scene_json["nodes"]

            node_index_to_parent_index = _get_node_index_to_parent_index(
                nodes_json=nodes_json, node_indices=scene_node_indices
            )
            mesh_index_to_node_indices = _get_mesh_index_to_node_indices(
                nodes_json=nodes_json, node_indices=scene_node_indices
            )
            transform_sequence = _get_transform_sequence(
                node_index_to_parent_index, mesh_index_to_node_indices
            )
            mesh_offsets = [
                transform_sequence.node_index_to_flattened_index[node_indices[0]]
                for node_indices in mesh_index_to_node_indices.values()
            ]

            self.scenes.append(
                GltfScene(
                    node_index_to_parent_index,
                    mesh_index_to_node_indices,
                    transform_sequence,
                    mesh_offsets,
                )
            )
