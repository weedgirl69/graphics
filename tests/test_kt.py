import array
import dataclasses
import itertools
import math
import pickle

from typing import Optional
import numpy as np
import png
import pytest
import kt
import kt.graphics_app
from kt import (
    BufferUsage,
    CommandBufferUsage,
    Filter,
    Format,
    ImageAspect,
    ImageLayout,
    ImageUsage,
    IndexType,
    LoadOp,
    StoreOp,
    VertexInputRate,
)
from kt.command_buffer_builder import CommandBufferBuilder
import kt.mesh


CUBE_MESH: Optional[kt.mesh.Mesh] = None
with open("tests/cube.pickle", "rb") as cube_file:
    CUBE_MESH = pickle.load(cube_file)


class AppTest:
    # pylint: disable=too-few-public-methods
    def __init__(
        self, name=None, width: int = 420, height: int = 420, max_error: float = 0.0
    ):
        self.name = name
        self.width = width
        self.height = height
        self.max_error = max_error

    def __call__(self, method):
        if not self.name:
            self.name = method.__name__

        @dataclasses.dataclass(frozen=True)
        class TestData:
            app: kt.graphics_app.GraphicsApp
            width: int
            height: int
            readback_buffer: kt.Buffer
            readback_buffer_memory: kt.vk.ffi.buffer
            command_pool: kt.CommandPool
            command_buffer: kt.CommandBuffer

        def wrapper():
            # pylint: disable=no-member
            try:
                with kt.graphics_app.run_graphics() as app:

                    readback_buffer = app.new_buffer(
                        byte_count=self.width * self.height * 4,
                        usage=BufferUsage.TRANSFER_DESTINATION,
                    )
                    mapped_memory = app.new_memory_set(downloadable=[readback_buffer])
                    readback_buffer_memory = mapped_memory[readback_buffer]
                    command_pool = app.new_command_pool()
                    command_buffer = app.allocate_command_buffer(command_pool)

                    method(
                        app,
                        TestData(
                            app,
                            self.width,
                            self.height,
                            readback_buffer,
                            readback_buffer_memory,
                            command_pool,
                            command_buffer,
                        ),
                    )

                    app.graphics_queue.submit(command_buffer)
                    app.graphics_queue.wait()

                    test_image_bytes = readback_buffer_memory[
                        0 : self.width * self.height * 4
                    ]

                    golden_path = "tests/goldens/" + self.name[len("test_") :] + ".png"
                    try:
                        with open(golden_path, "rb") as file:
                            png_reader = png.Reader(file=file)
                            golden_image_bytes = png_reader.read_flat()[2].tobytes()
                            mean_squared_error = sum(
                                (a - b) ** 2
                                for a, b in zip(golden_image_bytes, test_image_bytes)
                            ) / len(golden_image_bytes)
                            assert mean_squared_error <= self.max_error
                    except FileNotFoundError:
                        with open(golden_path, "wb") as file:
                            png_writer = png.Writer(self.width, self.height, alpha=True)
                            png_writer.write_array(file, test_image_bytes)
            finally:
                if app.errors:
                    import pprint

                    pprint.pprint(app.errors)
                    for error in app.errors:
                        print(error)
                    assert False

        return wrapper


@AppTest(max_error=0.0005)
def test_metallic_roughness(app, test_data):
    with open("tests/lighting_mesh.pickle", "rb") as file:
        lighting_mesh = pickle.load(file)
        sample_count = 3
        color_target_image = app.new_image(
            format=Format.R8G8B8A8_SRGB,
            usage=ImageUsage.COLOR_ATTACHMENT,
            width=test_data.width * 2,
            height=test_data.height * 2,
            sample_count=sample_count,
        )
        depth_target_image = app.new_image(
            format=Format.D24X8,
            usage=ImageUsage.DEPTH_ATTACHMENT,
            width=test_data.width * 2,
            height=test_data.height * 2,
            sample_count=sample_count,
        )
        resolve_target_image = app.new_image(
            format=Format.R8G8B8A8_SRGB,
            usage=ImageUsage.COLOR_ATTACHMENT | ImageUsage.TRANSFER_SOURCE,
            width=test_data.width * 2,
            height=test_data.height * 2,
        )
        downsampled_target_image = app.new_image(
            format=Format.R8G8B8A8_SRGB,
            usage=ImageUsage.TRANSFER_DESTINATION | ImageUsage.TRANSFER_SOURCE,
            width=test_data.width,
            height=test_data.height,
        )

        index_buffer = app.new_buffer(
            byte_count=len(lighting_mesh.indices.tobytes()), usage=BufferUsage.INDEX
        )
        positions_buffer = app.new_buffer(
            byte_count=len(lighting_mesh.positions.tobytes()), usage=BufferUsage.VERTEX
        )
        normals_buffer = app.new_buffer(
            byte_count=len(lighting_mesh.normals.tobytes()), usage=BufferUsage.VERTEX
        )
        app.new_memory_set(
            device_optimal=[resolve_target_image, downsampled_target_image],
            lazily_allocated=[color_target_image, depth_target_image],
            uploadable=[index_buffer, positions_buffer, normals_buffer],
            initial_values={
                index_buffer: lighting_mesh.indices.tobytes(),
                positions_buffer: lighting_mesh.positions.tobytes(),
                normals_buffer: lighting_mesh.normals.tobytes(),
            },
        )
        render_pass = app.new_render_pass(
            attachments=[
                kt.new_attachment_description(
                    pixel_format=Format.R8G8B8A8_SRGB,
                    load_op=LoadOp.CLEAR,
                    store_op=StoreOp.DISCARD,
                    sample_count=sample_count,
                ),
                kt.new_attachment_description(
                    pixel_format=Format.D24X8,
                    load_op=LoadOp.CLEAR,
                    store_op=StoreOp.DISCARD,
                    sample_count=sample_count,
                ),
                kt.new_attachment_description(
                    pixel_format=Format.R8G8B8A8_SRGB,
                    load_op=LoadOp.DONT_CARE,
                    store_op=StoreOp.STORE,
                    final_layout=ImageLayout.TRANSFER_SOURCE,
                ),
            ],
            subpass_descriptions=[
                kt.new_subpass_description(
                    color_attachments=[(0, ImageLayout.COLOR)],
                    resolve_attachments=[(2, ImageLayout.TRANSFER_DESTINATION)],
                    depth_attachment=1,
                )
            ],
        )
        color_target_view = app.new_image_view(
            image=color_target_image,
            format=Format.R8G8B8A8_SRGB,
            aspect=ImageAspect.COLOR,
        )
        depth_target_view = app.new_image_view(
            image=depth_target_image, format=Format.D24X8, aspect=ImageAspect.DEPTH
        )
        resolve_target_view = app.new_image_view(
            image=resolve_target_image,
            format=Format.R8G8B8A8_SRGB,
            aspect=ImageAspect.COLOR,
        )
        framebuffer = app.new_framebuffer(
            render_pass=render_pass,
            attachments=[color_target_view, depth_target_view, resolve_target_view],
            width=test_data.width * 2,
            height=test_data.height * 2,
            layers=1,
        )
        shader_set = app.new_shader_set(
            "tests/position_normal.vert.glsl", "tests/metallic_roughness.frag.glsl"
        )
        pipeline_layout = app.new_pipeline_layout()
        pipeline = app.new_pipeline(
            kt.new_graphics_pipeline_description(
                pipeline_layout=pipeline_layout,
                render_pass=render_pass,
                vertex_shader=shader_set.position_normal_vert,
                fragment_shader=shader_set.metallic_roughness_frag,
                vertex_attributes=[
                    kt.new_vertex_attribute(
                        location=0, binding=0, pixel_format=Format.R32G32B32_FLOAT
                    ),
                    kt.new_vertex_attribute(
                        location=1, binding=1, pixel_format=Format.R32G32B32_FLOAT
                    ),
                ],
                vertex_bindings=[
                    kt.new_vertex_binding(binding=0, stride=12),
                    kt.new_vertex_binding(binding=1, stride=12),
                ],
                multisample_description=kt.new_multisample_description(
                    sample_count=sample_count
                ),
                depth_description=kt.new_depth_description(
                    test_enabled=True, write_enabled=True
                ),
                width=test_data.width * 2,
                height=test_data.height * 2,
            )
        )

        with CommandBufferBuilder(
            command_buffer=test_data.command_buffer,
            usage=CommandBufferUsage.ONE_TIME_SUBMIT,
        ) as command_buffer_builder:
            command_buffer_builder.begin_render_pass(
                render_pass=render_pass,
                framebuffer=framebuffer,
                width=test_data.width * 2,
                height=test_data.height * 2,
                clear_values=[
                    kt.new_clear_value(color=(0.5, 0.5, 0.5, 1.0)),
                    kt.new_clear_value(depth=1),
                ],
            )

            command_buffer_builder.bind_pipeline(pipeline)

            command_buffer_builder.bind_index_buffer(
                buffer=index_buffer, index_type=IndexType.UINT16
            )
            command_buffer_builder.bind_vertex_buffers(
                [positions_buffer, normals_buffer]
            )
            command_buffer_builder.draw_indexed(index_count=len(lighting_mesh.indices))

            command_buffer_builder.end_render_pass()

            command_buffer_builder.pipeline_barrier(
                image=downsampled_target_image,
                new_layout=ImageLayout.TRANSFER_DESTINATION,
            )

            command_buffer_builder.blit_image(
                source_image=resolve_target_image,
                source_width=test_data.width * 2,
                source_height=test_data.height * 2,
                destination_image=downsampled_target_image,
                destination_width=test_data.width,
                destination_height=test_data.height,
            )

            command_buffer_builder.pipeline_barrier(
                image=downsampled_target_image,
                mip_count=1,
                old_layout=ImageLayout.TRANSFER_DESTINATION,
                new_layout=ImageLayout.TRANSFER_SOURCE,
            )

            command_buffer_builder.copy_image_to_buffer(
                image=downsampled_target_image,
                buffer=test_data.readback_buffer,
                width=test_data.width,
                height=test_data.height,
            )


@AppTest(width=480, height=320)
def test_texture(app, test_data):
    color_target_image = app.new_image(
        format=Format.R8G8B8A8_SRGB,
        usage=ImageUsage.COLOR_ATTACHMENT | ImageUsage.TRANSFER_SOURCE,
        width=test_data.width,
        height=test_data.height,
    )
    texture_size = 160
    texture_mip_count = 1 + int(math.log2(texture_size))
    test_texture_image = app.new_image(
        format=Format.R8G8B8A8_SRGB,
        usage=ImageUsage.SAMPLED
        | ImageUsage.TRANSFER_DESTINATION
        | ImageUsage.TRANSFER_SOURCE,
        width=texture_size,
        height=texture_size,
        mip_count=texture_mip_count,
    )
    test_texture_buffer_byte_count = texture_size * texture_size * 4
    test_texture_buffer = app.new_buffer(
        byte_count=test_texture_buffer_byte_count, usage=BufferUsage.TRANSFER_SOURCE
    )

    def create_square(x, y, size):
        # pylint: disable=invalid-name
        return (x, y), (x, y + size), (x + size, y + size), (x + size, y)

    positions = [
        *create_square(0, 0, texture_size * 2),
        *create_square(texture_size * 2, 0, texture_size),
    ]
    x_offset = texture_size * 2
    current_size = texture_size // 2
    for _ in range(2, texture_mip_count + 1):
        positions += create_square(x_offset, texture_size, current_size)
        x_offset += current_size
        current_size //= 2

    texcoord_bytes = array.array(
        "f",
        itertools.chain.from_iterable(
            ((0, 0), (0, 1), (1, 1), (1, 0)) * (len(positions) // 4)
        ),
    ).tobytes()
    indices_bytes = array.array(
        "H",
        itertools.chain.from_iterable(
            (
                (i + 0, i + 1, i + 2, i + 2, i + 3, i + 0)
                for i in range(len(positions))[::4]
            )
        ),
    ).tobytes()
    positions_bytes = array.array(
        "f",
        itertools.chain.from_iterable(
            ((x / 480 * 2 - 1, y / 320 * 2 - 1) for x, y in positions)
        ),
    ).tobytes()
    index_buffer = app.new_buffer(
        byte_count=len(indices_bytes), usage=BufferUsage.INDEX
    )
    positions_buffer = app.new_buffer(
        byte_count=len(positions_bytes), usage=BufferUsage.VERTEX
    )
    texcoords_buffer = app.new_buffer(
        byte_count=len(texcoord_bytes), usage=BufferUsage.VERTEX
    )

    with open("tests/test_texture.png", "rb") as file:
        png_reader = png.Reader(file=file)
        app.new_memory_set(
            device_optimal=[color_target_image, test_texture_image],
            uploadable=[
                test_texture_buffer,
                index_buffer,
                positions_buffer,
                texcoords_buffer,
            ],
            initial_values={
                test_texture_buffer: bytes(
                    itertools.chain.from_iterable(png_reader.asRGBA8()[2])
                ),
                index_buffer: indices_bytes,
                positions_buffer: positions_bytes,
                texcoords_buffer: texcoord_bytes,
            },
        )
    test_texture_view = app.new_image_view(
        image=test_texture_image,
        format=Format.R8G8B8A8_SRGB,
        mip_count=texture_mip_count,
    )
    sampler = app.new_sampler(min_filter=Filter.LINEAR, mag_filter=Filter.LINEAR)
    descriptor_set_layout = app.new_descriptor_set_layout(
        [
            kt.new_descriptor_layout_binding(
                binding=0,
                stage=kt.ShaderStage.FRAGMENT,
                descriptor_type=kt.DescriptorType.COMBINED_IMAGE_SAMPLER,
                immutable_samplers=[sampler],
            )
        ]
    )
    descriptor_pool = app.new_descriptor_pool(
        max_set_count=1,
        descriptor_type_counts={kt.DescriptorType.COMBINED_IMAGE_SAMPLER: 1},
    )
    test_texture_descriptor_set = app.allocate_descriptor_sets(
        descriptor_pool=descriptor_pool, descriptor_set_layouts=[descriptor_set_layout]
    )[0]
    app.update_descriptor_sets(
        image_writes=[
            kt.DescriptorImageWrites(
                binding=0,
                image_infos=[
                    kt.DescriptorImageInfo(
                        image_view=test_texture_view, layout=kt.ImageLayout.SHADER
                    )
                ],
                count=1,
                descriptor_set=test_texture_descriptor_set,
                descriptor_type=kt.DescriptorType.COMBINED_IMAGE_SAMPLER,
            )
        ]
    )

    render_pass = app.new_render_pass(
        attachments=[
            kt.new_attachment_description(
                pixel_format=Format.R8G8B8A8_SRGB,
                load_op=LoadOp.CLEAR,
                store_op=StoreOp.STORE,
                final_layout=ImageLayout.TRANSFER_SOURCE,
            )
        ],
        subpass_descriptions=[
            kt.new_subpass_description(color_attachments=[(0, ImageLayout.COLOR)])
        ],
    )
    color_target_view = app.new_image_view(
        image=color_target_image, format=Format.R8G8B8A8_SRGB
    )
    framebuffer = app.new_framebuffer(
        render_pass=render_pass,
        attachments=[color_target_view],
        width=test_data.width,
        height=test_data.height,
        layers=1,
    )
    pipeline_layout = app.new_pipeline_layout([descriptor_set_layout])
    shader_set = app.new_shader_set(
        "tests/texture_test.vert.glsl", "tests/texture_test.frag.glsl"
    )
    pipeline_description = kt.new_graphics_pipeline_description(
        pipeline_layout=pipeline_layout,
        render_pass=render_pass,
        vertex_shader=shader_set.texture_test_vert,
        fragment_shader=shader_set.texture_test_frag,
        vertex_attributes=[
            kt.new_vertex_attribute(
                location=0, binding=0, pixel_format=Format.R32G32_FLOAT
            ),
            kt.new_vertex_attribute(
                location=1, binding=1, pixel_format=Format.R32G32_FLOAT
            ),
        ],
        vertex_bindings=[
            kt.new_vertex_binding(binding=0, stride=8),
            kt.new_vertex_binding(binding=1, stride=8),
        ],
        width=test_data.width,
        height=test_data.height,
    )
    pipeline = app.new_pipeline(pipeline_description)

    with CommandBufferBuilder(
        command_buffer=test_data.command_buffer,
        usage=CommandBufferUsage.ONE_TIME_SUBMIT,
    ) as command_buffer_builder:
        command_buffer_builder.pipeline_barrier(
            image=test_texture_image,
            mip_count=texture_mip_count,
            old_layout=ImageLayout.UNDEFINED,
            new_layout=ImageLayout.TRANSFER_DESTINATION,
        )

        command_buffer_builder.copy_buffer_to_image(
            buffer=test_texture_buffer,
            image=test_texture_image,
            width=texture_size,
            height=texture_size,
        )

        for source_level in range(texture_mip_count - 1):
            destination_level = source_level + 1
            command_buffer_builder.pipeline_barrier(
                image=test_texture_image,
                base_mip_level=source_level,
                mip_count=1,
                old_layout=ImageLayout.TRANSFER_DESTINATION,
                new_layout=ImageLayout.TRANSFER_SOURCE,
            )

            command_buffer_builder.blit_image(
                source_image=test_texture_image,
                source_subresource_index=source_level,
                source_width=texture_size >> source_level,
                source_height=texture_size >> source_level,
                destination_image=test_texture_image,
                destination_subresource_index=destination_level,
                destination_width=texture_size >> destination_level,
                destination_height=texture_size >> destination_level,
            )

        command_buffer_builder.pipeline_barrier(
            image=test_texture_image,
            base_mip_level=0,
            mip_count=texture_mip_count,
            old_layout=ImageLayout.UNDEFINED,
            new_layout=ImageLayout.SHADER,
        )
        command_buffer_builder.begin_render_pass(
            render_pass=render_pass,
            framebuffer=framebuffer,
            width=test_data.width,
            height=test_data.height,
            clear_values=[kt.new_clear_value(color=(0, 0, 0, 0))],
        )

        command_buffer_builder.bind_pipeline(pipeline)
        command_buffer_builder.bind_descriptor_sets(
            pipeline_layout=pipeline_layout,
            descriptor_sets=[test_texture_descriptor_set],
        )

        command_buffer_builder.bind_index_buffer(
            buffer=index_buffer, index_type=IndexType.UINT16
        )
        command_buffer_builder.bind_vertex_buffers([positions_buffer, texcoords_buffer])
        command_buffer_builder.draw_indexed(index_count=len(indices_bytes) // 2)

        command_buffer_builder.end_render_pass()

        command_buffer_builder.copy_image_to_buffer(
            image=color_target_image,
            buffer=test_data.readback_buffer,
            width=test_data.width,
            height=test_data.height,
        )


def cube_app_test(method):
    @AppTest(name=method.__name__)
    def wrapper(app, test_data):
        cube_resources = {
            "color_target": app.new_image(
                format=Format.R8G8B8A8_SRGB,
                usage=ImageUsage.COLOR_ATTACHMENT | ImageUsage.TRANSFER_SOURCE,
                width=test_data.width,
                height=test_data.height,
            ),
            "depth_target": app.new_image(
                format=Format.D24X8,
                usage=ImageUsage.DEPTH_ATTACHMENT,
                width=test_data.width,
                height=test_data.height,
            ),
            "index_buffer": app.new_buffer(
                byte_count=len(CUBE_MESH.indices.tobytes()), usage=BufferUsage.INDEX
            ),
            "positions_buffer": app.new_buffer(
                byte_count=len(CUBE_MESH.positions.tobytes()), usage=BufferUsage.VERTEX
            ),
            "normals_buffer": app.new_buffer(
                byte_count=len(CUBE_MESH.normals.tobytes()), usage=BufferUsage.VERTEX
            ),
            "render_pass": app.new_render_pass(
                attachments=[
                    kt.new_attachment_description(
                        pixel_format=Format.R8G8B8A8_SRGB,
                        load_op=LoadOp.CLEAR,
                        store_op=StoreOp.STORE,
                        final_layout=ImageLayout.TRANSFER_SOURCE,
                    ),
                    kt.new_attachment_description(
                        pixel_format=Format.D24X8,
                        load_op=LoadOp.CLEAR,
                        store_op=StoreOp.DISCARD,
                    ),
                ],
                subpass_descriptions=[
                    kt.new_subpass_description(
                        color_attachments=[(0, ImageLayout.COLOR)], depth_attachment=1
                    )
                ],
            ),
        }

        app.new_memory_set(
            device_optimal=[cube_resources["color_target"]],
            uploadable=[
                cube_resources["index_buffer"],
                cube_resources["positions_buffer"],
                cube_resources["normals_buffer"],
            ],
            lazily_allocated=[cube_resources["depth_target"]],
            initial_values={
                cube_resources["index_buffer"]: CUBE_MESH.indices.tobytes(),
                cube_resources["positions_buffer"]: CUBE_MESH.positions.tobytes(),
                cube_resources["normals_buffer"]: CUBE_MESH.normals.tobytes(),
            },
        )
        cube_resources.update(
            {
                "color_target_view": app.new_image_view(
                    image=cube_resources["color_target"],
                    format=Format.R8G8B8A8_SRGB,
                    aspect=ImageAspect.COLOR,
                ),
                "depth_target_view": app.new_image_view(
                    image=cube_resources["depth_target"],
                    format=Format.D24X8,
                    aspect=ImageAspect.DEPTH,
                ),
            }
        )

        cube_resources["framebuffer"] = app.new_framebuffer(
            render_pass=cube_resources["render_pass"],
            attachments=[
                cube_resources["color_target_view"],
                cube_resources["depth_target_view"],
            ],
            width=test_data.width,
            height=test_data.height,
            layers=1,
        )

        method(app, test_data, cube_resources)

    return wrapper


@cube_app_test
def test_instanced_cubes(app, test_data, cube_resources):
    # pylint: disable=too-many-locals
    instance_transforms = []
    instance_count = 2048
    for i in range(instance_count):
        theta = 2 * i / instance_count * math.pi
        forward = (-math.sin(theta), math.cos(theta), 0)
        right = (0, 0, -1)
        _up = np.cross(forward, right)
        right = np.multiply(-math.sin(theta), right) + np.multiply(math.cos(theta), _up)
        _up = np.cross(forward, right)
        instance_scale = (0.5 + 0.5 * math.sin(theta * 2)) * 0.2

        instance_transforms.extend(
            (
                (
                    instance_scale * right[0],
                    instance_scale * _up[0],
                    instance_scale * forward[0],
                    1.5 * math.cos(theta),
                ),
                (
                    instance_scale * right[1],
                    instance_scale * _up[1],
                    instance_scale * forward[1],
                    1.5 * math.sin(theta),
                ),
                (
                    instance_scale * right[2],
                    instance_scale * _up[2],
                    instance_scale * forward[2],
                    0,
                ),
            )
        )

    instance_transforms = array.array(
        "f", itertools.chain.from_iterable(instance_transforms)
    )
    instance_buffer = app.new_buffer(
        byte_count=len(instance_transforms.tobytes()), usage=BufferUsage.VERTEX
    )
    app.new_memory_set(
        uploadable=[instance_buffer],
        initial_values={instance_buffer: instance_transforms.tobytes()},
    )

    shader_set = app.new_shader_set(
        "tests/instanced_cube.vert.glsl", "tests/cube.frag.glsl"
    )

    float3_format = Format.R32G32B32_FLOAT
    float4_format = Format.R32G32B32A32_FLOAT
    pipeline = app.new_pipeline(
        kt.new_graphics_pipeline_description(
            pipeline_layout=app.new_pipeline_layout(),
            render_pass=cube_resources["render_pass"],
            vertex_shader=shader_set.instanced_cube_vert,
            fragment_shader=shader_set.cube_frag,
            vertex_attributes=[
                kt.new_vertex_attribute(
                    location=0, binding=0, pixel_format=float3_format
                ),
                kt.new_vertex_attribute(
                    location=1, binding=1, pixel_format=float3_format
                ),
                kt.new_vertex_attribute(
                    location=2, binding=2, pixel_format=float4_format, offset=0
                ),
                kt.new_vertex_attribute(
                    location=3, binding=2, pixel_format=float4_format, offset=16
                ),
                kt.new_vertex_attribute(
                    location=4, binding=2, pixel_format=float4_format, offset=32
                ),
            ],
            vertex_bindings=[
                kt.new_vertex_binding(binding=0, stride=12),
                kt.new_vertex_binding(binding=1, stride=12),
                kt.new_vertex_binding(
                    binding=2, stride=48, input_rate=VertexInputRate.PER_INSTANCE
                ),
            ],
            depth_description=kt.new_depth_description(
                test_enabled=True, write_enabled=True
            ),
            width=test_data.width,
            height=test_data.height,
        )
    )

    with CommandBufferBuilder(
        command_buffer=test_data.command_buffer,
        usage=CommandBufferUsage.ONE_TIME_SUBMIT,
    ) as command_buffer_builder:
        command_buffer_builder.begin_render_pass(
            render_pass=cube_resources["render_pass"],
            framebuffer=cube_resources["framebuffer"],
            width=test_data.width,
            height=test_data.height,
            clear_values=[
                kt.new_clear_value(color=(0.5, 0.5, 0.5, 1.0)),
                kt.new_clear_value(depth=1),
            ],
        )

        command_buffer_builder.bind_pipeline(pipeline)

        command_buffer_builder.bind_index_buffer(
            buffer=cube_resources["index_buffer"], index_type=IndexType.UINT16
        )
        command_buffer_builder.bind_vertex_buffers(
            [
                cube_resources["positions_buffer"],
                cube_resources["normals_buffer"],
                instance_buffer,
            ]
        )
        command_buffer_builder.draw_indexed(
            index_count=36, instance_count=instance_count
        )

        command_buffer_builder.end_render_pass()

        command_buffer_builder.copy_image_to_buffer(
            image=cube_resources["color_target"],
            buffer=test_data.readback_buffer,
            width=test_data.width,
            height=test_data.height,
        )


@cube_app_test
def test_cube(app, test_data, cube_resources):
    shader_set = app.new_shader_set("tests/cube.vert.glsl", "tests/cube.frag.glsl")

    pipeline = app.new_pipeline(
        kt.new_graphics_pipeline_description(
            pipeline_layout=app.new_pipeline_layout(),
            render_pass=cube_resources["render_pass"],
            vertex_shader=shader_set.cube_vert,
            fragment_shader=shader_set.cube_frag,
            vertex_attributes=[
                kt.new_vertex_attribute(
                    location=0, binding=0, pixel_format=Format.R32G32B32_FLOAT
                ),
                kt.new_vertex_attribute(
                    location=1, binding=1, pixel_format=Format.R32G32B32_FLOAT
                ),
            ],
            vertex_bindings=[
                kt.new_vertex_binding(binding=0, stride=12),
                kt.new_vertex_binding(binding=1, stride=12),
            ],
            width=test_data.width,
            height=test_data.height,
            depth_description=kt.new_depth_description(
                test_enabled=False, write_enabled=False
            ),
        )
    )

    with CommandBufferBuilder(
        command_buffer=test_data.command_buffer,
        usage=CommandBufferUsage.ONE_TIME_SUBMIT,
    ) as command_buffer_builder:
        command_buffer_builder.begin_render_pass(
            render_pass=cube_resources["render_pass"],
            framebuffer=cube_resources["framebuffer"],
            width=test_data.width,
            height=test_data.height,
            clear_values=[
                kt.new_clear_value(color=(0.5, 0.5, 0.5, 1.0)),
                kt.new_clear_value(depth=1),
            ],
        )

        command_buffer_builder.bind_pipeline(pipeline)

        command_buffer_builder.bind_index_buffer(
            buffer=cube_resources["index_buffer"], index_type=IndexType.UINT16
        )
        command_buffer_builder.bind_vertex_buffers(
            [cube_resources["positions_buffer"], cube_resources["normals_buffer"]]
        )
        command_buffer_builder.draw_indexed(index_count=36)

        command_buffer_builder.end_render_pass()

        command_buffer_builder.copy_image_to_buffer(
            image=cube_resources["color_target"],
            buffer=test_data.readback_buffer,
            width=test_data.width,
            height=test_data.height,
        )


@AppTest()
def test_triangle(app, test_data):
    image = app.new_image(
        format=Format.R8G8B8A8_SRGB,
        usage=ImageUsage.COLOR_ATTACHMENT | ImageUsage.TRANSFER_SOURCE,
        width=test_data.width,
        height=test_data.height,
    )
    app.new_memory_set(device_optimal=[image])
    render_pass = app.new_render_pass(
        attachments=[
            kt.new_attachment_description(
                pixel_format=Format.R8G8B8A8_SRGB,
                load_op=LoadOp.CLEAR,
                store_op=StoreOp.STORE,
                final_layout=ImageLayout.TRANSFER_SOURCE,
            )
        ],
        subpass_descriptions=[
            kt.new_subpass_description(color_attachments=[(0, ImageLayout.COLOR)])
        ],
    )
    image_view = app.new_image_view(image=image, format=Format.R8G8B8A8_SRGB)
    framebuffer = app.new_framebuffer(
        render_pass=render_pass,
        attachments=[image_view],
        width=test_data.width,
        height=test_data.height,
        layers=1,
    )
    pipeline_layout = app.new_pipeline_layout()
    shader_set = app.new_shader_set(
        "tests/triangle.vert.glsl", "tests/triangle.frag.glsl"
    )
    pipeline_description = kt.new_graphics_pipeline_description(
        pipeline_layout=pipeline_layout,
        render_pass=render_pass,
        vertex_shader=shader_set.triangle_vert,
        fragment_shader=shader_set.triangle_frag,
        width=test_data.width,
        height=test_data.height,
    )
    pipeline = app.new_pipeline(pipeline_description)

    with CommandBufferBuilder(
        command_buffer=test_data.command_buffer,
        usage=CommandBufferUsage.ONE_TIME_SUBMIT,
    ) as command_buffer_builder:
        command_buffer_builder.begin_render_pass(
            render_pass=render_pass,
            framebuffer=framebuffer,
            width=test_data.width,
            height=test_data.height,
            clear_values=[kt.new_clear_value(color=(0, 0, 0, 0))],
        )

        command_buffer_builder.bind_pipeline(pipeline)

        command_buffer_builder.draw(vertex_count=3, instance_count=1)

        command_buffer_builder.end_render_pass()

        command_buffer_builder.copy_image_to_buffer(
            image=image,
            buffer=test_data.readback_buffer,
            width=test_data.width,
            height=test_data.height,
        )


@AppTest()
def test_clear(app, test_data):
    clear_color = [5 / 255, 144 / 255, 51 / 255, 1.0]
    image = app.new_image(
        format=Format.R8G8B8A8_UNORM,
        usage=ImageUsage.TRANSFER_DESTINATION | ImageUsage.TRANSFER_SOURCE,
        width=test_data.width,
        height=test_data.height,
    )
    app.new_memory_set(device_optimal=[image])

    with CommandBufferBuilder(
        command_buffer=test_data.command_buffer,
        usage=CommandBufferUsage.ONE_TIME_SUBMIT,
    ) as command_buffer_builder:
        command_buffer_builder.pipeline_barrier(
            image=image, new_layout=ImageLayout.TRANSFER_DESTINATION
        )

        command_buffer_builder.clear_color_image(image=image, color=clear_color)

        command_buffer_builder.pipeline_barrier(
            image=image,
            old_layout=ImageLayout.TRANSFER_DESTINATION,
            new_layout=ImageLayout.TRANSFER_SOURCE,
        )

        command_buffer_builder.copy_image_to_buffer(
            image=image,
            width=test_data.width,
            height=test_data.height,
            buffer=test_data.readback_buffer,
        )


def test_errors():
    with kt.graphics_app.run_graphics() as app:
        buffer = app.new_buffer(byte_count=0, usage=0)
        with pytest.raises(kt.vk.VkErrorOutOfDeviceMemory):
            app.new_memory_set(device_optimal=[buffer])
    assert app.errors
