from __future__ import annotations
import array
import base64
import collections
import json
import dataclasses
import typing

AffineTransform = typing.Tuple[
    float, float, float, float, float, float, float, float, float, float, float, float
]


@dataclasses.dataclass(frozen=True)
class Bounds:
    min: typing.Tuple[float, float, float]
    max: typing.Tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class IndexData:
    byte_offset: int
    index_size: int


@dataclasses.dataclass(frozen=True)
class Primitive:
    bounds: Bounds
    count: int
    index_data: typing.Optional[IndexData]
    positions_byte_offset: int
    normals_byte_offset: typing.Optional[int]


@dataclasses.dataclass(frozen=True)
class TransformSequence:
    node_index_to_flattened_index: typing.Dict[int, int]
    transform_source_index_to_destination_index: typing.List[typing.Tuple[int, int]]


@dataclasses.dataclass(frozen=True)
class Scene:
    mesh_index_to_node_indices: typing.List[typing.List[int]]
    transform_sequence: TransformSequence
    mesh_index_to_base_instance_offset: typing.List[int]


@dataclasses.dataclass(frozen=True)
class Model:
    attributes_bytes: bytes
    indices_bytes: bytes
    meshes: typing.List[typing.List[Primitive]]
    node_transforms: typing.List[AffineTransform]
    scenes: typing.List[Scene]


def _get_node_transforms(gltf_json: typing.Dict) -> typing.List[AffineTransform]:
    def get_transform(node_json: typing.Dict) -> AffineTransform:
        if "matrix" in node_json:
            matrix = [float(_) for _ in node_json["matrix"]]
            return (
                matrix[0],
                matrix[4],
                matrix[8],
                matrix[12],
                matrix[1],
                matrix[5],
                matrix[9],
                matrix[13],
                matrix[2],
                matrix[6],
                matrix[10],
                matrix[14],
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
            scale_y * m10,
            scale_z * m20,
            translation_x,
            scale_x * m01,
            scale_y * m11,
            scale_z * m21,
            translation_y,
            scale_x * m02,
            scale_y * m12,
            scale_z * m22,
            translation_z,
        )

    return [get_transform(node_json) for node_json in gltf_json["nodes"]]


def _get_node_index_to_parent_index(
    *,
    nodes_json: typing.List[typing.Dict],
    node_indices: typing.List[int],
    parent_index: typing.Optional[int] = None,
    node_index_to_parent_index: typing.List[typing.Optional[int]] = None,
) -> typing.List[typing.Optional[int]]:
    if node_index_to_parent_index is None:
        node_index_to_parent_index = [None] * len(nodes_json)

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
    mesh_index_to_node_indices: typing.List[typing.List[int]],
) -> typing.List[typing.List[int]]:

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
    node_index_to_parent_index: typing.List[typing.Optional[int]],
    mesh_index_to_node_indices: typing.List[typing.List[int]],
) -> TransformSequence:
    node_index_to_flattened_index = {}
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

    for node_indices in mesh_index_to_node_indices:
        for node_index in node_indices:
            node_index_to_flattened_index[node_index] = current_instance_index
            current_instance_index += 1

    for node_indices in mesh_index_to_node_indices:
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
            byte_count = count * byte_stride

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
    *,
    accessors: typing.List[bytes],
    accessors_json: typing.List[typing.Dict],
    meshes_json: typing.List[typing.Dict],
) -> typing.Tuple[typing.List[typing.List[Primitive]], bytes, bytes]:
    indices_bytes = bytearray()
    attributes_bytes = bytearray()
    indices_accessor_index_to_upsampled_indices_offset = {}

    def get_index_data_and_count(
        indices_accessor_index: int
    ) -> typing.Tuple[IndexData, int]:
        byte_offset = len(indices_bytes)
        index_size = 2
        accessor_json = accessors_json[indices_accessor_index]
        index_component_type = accessor_json["componentType"]
        if index_component_type == 5121:
            if (
                indices_accessor_index
                in indices_accessor_index_to_upsampled_indices_offset
            ):
                byte_offset = indices_accessor_index_to_upsampled_indices_offset[
                    indices_accessor_index
                ]
            else:
                indices_bytes.extend(
                    array.array(
                        "H", list(index for index in accessors[indices_accessor_index])
                    )
                )
        else:
            if index_component_type == 5125:
                index_size = 4
            indices_bytes.extend(accessors[indices_accessor_index])
        return (
            IndexData(byte_offset=byte_offset, index_size=index_size),
            accessor_json["count"],
        )

    def create_gltf_primitive(primitive_json: typing.Dict):
        attributes_json = primitive_json["attributes"]
        positions_accessor_index = attributes_json["POSITION"]
        positions_accessor_json = accessors_json[positions_accessor_index]

        accessor_min = positions_accessor_json["min"]
        accessor_max = positions_accessor_json["max"]
        bounds = Bounds(
            min=(accessor_min[0], accessor_min[1], accessor_min[2]),
            max=(accessor_max[0], accessor_max[1], accessor_max[2]),
        )

        index_data = None
        if "indices" in primitive_json:
            index_data, count = get_index_data_and_count(primitive_json["indices"])
        else:
            count = positions_accessor_json["count"]

        positions_byte_offset = len(attributes_bytes)
        attributes_bytes.extend(accessors[positions_accessor_index])

        normals_byte_offset = None
        if "NORMAL" in attributes_json:
            normals_byte_offset = len(attributes_bytes)
            attributes_bytes.extend(accessors[attributes_json["NORMAL"]])

        return Primitive(
            bounds=bounds,
            count=count,
            index_data=index_data,
            positions_byte_offset=positions_byte_offset,
            normals_byte_offset=normals_byte_offset,
        )

    return (
        [
            [
                create_gltf_primitive(primitive_json)
                for primitive_json in mesh_json["primitives"]
            ]
            for mesh_json in meshes_json
        ],
        indices_bytes,
        attributes_bytes,
    )


def from_json(file: typing.TextIO, uri_resolver):
    gltf_json = json.load(file)

    node_transforms = _get_node_transforms(gltf_json)

    accessors = _get_accessors(
        accessors_json=gltf_json["accessors"],
        buffers_json=gltf_json["buffers"],
        buffer_views_json=gltf_json["bufferViews"],
        uri_resolver=uri_resolver,
    )

    meshes, indices_bytes, attributes_bytes = _get_mesh_index_to_primitives(
        accessors=accessors,
        accessors_json=gltf_json["accessors"],
        meshes_json=gltf_json["meshes"],
    )

    nodes_json = gltf_json["nodes"]
    scenes: typing.List[Scene] = list()
    for scene_json in gltf_json["scenes"]:
        scene_node_indices = scene_json["nodes"]

        node_index_to_parent_index = _get_node_index_to_parent_index(
            nodes_json=nodes_json, node_indices=scene_node_indices
        )
        mesh_index_to_node_indices = _get_mesh_index_to_node_indices(
            nodes_json=nodes_json,
            node_indices=scene_node_indices,
            mesh_index_to_node_indices=[list() for _ in meshes],
        )
        transform_sequence = _get_transform_sequence(
            node_index_to_parent_index, mesh_index_to_node_indices
        )
        mesh_index_to_base_instance_offset = [
            transform_sequence.node_index_to_flattened_index[node_indices[0]]
            for node_indices in mesh_index_to_node_indices
        ]

        scenes.append(
            Scene(
                mesh_index_to_node_indices,
                transform_sequence,
                mesh_index_to_base_instance_offset,
            )
        )

    return Model(
        attributes_bytes=attributes_bytes,
        indices_bytes=indices_bytes,
        node_transforms=node_transforms,
        meshes=meshes,
        scenes=scenes,
    )
