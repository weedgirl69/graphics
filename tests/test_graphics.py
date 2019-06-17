import array
import itertools
import math
import numpy as np
import pytest
import png
import graphics
from graphics.types import (
    AttachmentDescription,
    BufferUsage,
    ClearValue,
    CommandBufferUsage,
    DepthDescription,
    Format,
    GraphicsPipelineDescription,
    ImageAspect,
    ImageLayout,
    ImageUsage,
    IndexType,
    LoadOp,
    SubpassDescription,
    StoreOp,
    VertexAttribute,
    VertexBinding,
    VertexInputRate,
)

CUBE_INDICES = array.array(
    "H",
    itertools.chain.from_iterable(
        ((i + 0, i + 1, i + 2, i + 2, i + 3, i + 0) for i in range(24)[::4])
    ),
)

CUBE_POSITIONS = array.array(
    "f",
    itertools.chain(
        (-1, 1, 1),
        (-1, 1, -1),
        (-1, -1, -1),
        (-1, -1, 1),
        (1, 1, -1),
        (1, 1, 1),
        (1, -1, 1),
        (1, -1, -1),
        (-1, -1, -1),
        (1, -1, -1),
        (1, -1, 1),
        (-1, -1, 1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, 1, 1),
        (-1, 1, -1),
        (1, 1, -1),
        (1, -1, -1),
        (-1, -1, -1),
        (1, 1, 1),
        (-1, 1, 1),
        (-1, -1, 1),
        (1, -1, 1),
    ),
)


CUBE_NORMALS = array.array(
    "f",
    itertools.chain(
        (-1, 0, 0) * 4,
        (1, 0, 0) * 4,
        (0, -1, 0) * 4,
        (0, 1, 0) * 4,
        (0, 0, -1) * 4,
        (0, 0, 1) * 4,
    ),
)


@pytest.fixture(name="test_app")
def test_app_fixture(request):
    class TestData:
        # pylint: disable=too-few-public-methods
        # pylint: disable=too-many-instance-attributes
        def __init__(self, app):
            self.app = app
            self.golden_path = (
                "tests/goldens/" + request.function.__name__[len("test_") :] + ".png"
            )
            self.width = 420
            self.height = 420
            self.readback_buffer = app.new_buffer(
                byte_count=self.width * self.height * 4,
                usage=BufferUsage.TRANSFER_DESTINATION,
            )
            mapped_memory = app.new_memory_set(
                {self.readback_buffer: graphics.MemoryType.Downloadable}
            )
            self.readback_buffer_memory = mapped_memory[self.readback_buffer]
            self.command_pool = app.new_command_pool()
            self.command_buffer = app.allocate_command_buffer(self.command_pool)

    with graphics.App() as app:
        test_data = TestData(app)
        yield test_data, app

        app.graphics_queue.submit(test_data.command_buffer)
        app.graphics_queue.wait()

        test_image_bytes = test_data.readback_buffer_memory[
            0 : test_data.width * test_data.height * 4
        ]

        try:
            with open(test_data.golden_path, "rb") as file:
                png_reader = png.Reader(file=file)
                golden_image_bytes = png_reader.read_flat()[2].tobytes()
                assert golden_image_bytes == test_image_bytes
        except FileNotFoundError:
            with open(test_data.golden_path, "wb") as file:
                png_writer = png.Writer(test_data.width, test_data.height, alpha=True)
                png_writer.write_array(file, test_image_bytes)


def test_triangle(test_app):
    test_data, app = test_app
    image = app.new_render_target(
        format=Format.R8G8B8A8_SRGB,
        usage=ImageUsage.COLOR | ImageUsage.TRANSFER_SOURCE,
        width=test_data.width,
        height=test_data.height,
    )
    app.new_memory_set({image: graphics.MemoryType.DeviceOptimal})
    render_pass = app.new_render_pass(
        attachments=[
            AttachmentDescription(
                format=Format.R8G8B8A8_SRGB,
                load_op=LoadOp.CLEAR,
                store_op=StoreOp.STORE,
                final_layout=ImageLayout.TRANSFER_SOURCE,
            )
        ],
        subpass_descriptions=[
            SubpassDescription(color_attachments=[(0, ImageLayout.COLOR)])
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
    pipeline_description = GraphicsPipelineDescription(
        pipeline_layout=pipeline_layout,
        render_pass=render_pass,
        vertex_shader=shader_set.triangle_vert,
        fragment_shader=shader_set.triangle_frag,
        width=test_data.width,
        height=test_data.height,
    )
    pipeline = app.new_pipeline(pipeline_description)

    with graphics.CommandBufferBuilder(
        test_data.command_buffer, CommandBufferUsage.ONE_TIME_SUBMIT
    ) as command_buffer_builder:
        command_buffer_builder.begin_render_pass(
            render_pass=render_pass,
            framebuffer=framebuffer,
            width=test_data.width,
            height=test_data.height,
            clear_values=[ClearValue(color=(0, 0, 0, 0))],
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


def test_clear(test_app):
    test_data, app = test_app
    clear_color = [5 / 255, 144 / 255, 51 / 255, 1.0]
    image = app.new_render_target(
        format=Format.R8G8B8A8_UNORM,
        usage=ImageUsage.TRANSFER_DESTINATION | ImageUsage.TRANSFER_SOURCE,
        width=test_data.width,
        height=test_data.height,
    )
    app.new_memory_set({image: graphics.MemoryType.DeviceOptimal})

    with graphics.CommandBufferBuilder(
        test_data.command_buffer, CommandBufferUsage.ONE_TIME_SUBMIT
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


@pytest.fixture(name="test_cube_app")
def test_cube_app_fixture(test_app):
    test_data, app = test_app
    cube_resources = {
        "color_target": app.new_render_target(
            format=Format.R8G8B8A8_SRGB,
            usage=ImageUsage.COLOR | ImageUsage.TRANSFER_SOURCE,
            width=test_data.width,
            height=test_data.height,
        ),
        "depth_target": app.new_render_target(
            format=Format.D24X8,
            usage=ImageUsage.DEPTH,
            width=test_data.width,
            height=test_data.height,
        ),
        "index_buffer": app.new_buffer(
            byte_count=len(CUBE_INDICES.tobytes()), usage=BufferUsage.INDEX
        ),
        "positions_buffer": app.new_buffer(
            byte_count=len(CUBE_POSITIONS.tobytes()), usage=BufferUsage.VERTEX
        ),
        "normals_buffer": app.new_buffer(
            byte_count=len(CUBE_NORMALS.tobytes()), usage=BufferUsage.VERTEX
        ),
        "render_pass": app.new_render_pass(
            attachments=[
                AttachmentDescription(
                    format=Format.R8G8B8A8_SRGB,
                    load_op=LoadOp.CLEAR,
                    store_op=StoreOp.STORE,
                    final_layout=ImageLayout.TRANSFER_SOURCE,
                ),
                AttachmentDescription(
                    format=Format.D24X8, load_op=LoadOp.CLEAR, store_op=StoreOp.DISCARD
                ),
            ],
            subpass_descriptions=[
                SubpassDescription(
                    color_attachments=[(0, ImageLayout.COLOR)], depth_attachment=1
                )
            ],
        ),
    }

    app.new_memory_set(
        {
            cube_resources["color_target"]: graphics.MemoryType.DeviceOptimal,
            cube_resources["depth_target"]: graphics.MemoryType.LazilyAllocated,
            cube_resources["index_buffer"]: graphics.MemoryType.Uploadable,
            cube_resources["positions_buffer"]: graphics.MemoryType.Uploadable,
            cube_resources["normals_buffer"]: graphics.MemoryType.Uploadable,
        },
        initial_values={
            cube_resources["index_buffer"]: CUBE_INDICES.tobytes(),
            cube_resources["positions_buffer"]: CUBE_POSITIONS.tobytes(),
            cube_resources["normals_buffer"]: CUBE_NORMALS.tobytes(),
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

    return test_data, app, cube_resources


def test_instanced_cubes(test_cube_app):
    test_app, app, cube_resources = test_cube_app

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
        {instance_buffer: graphics.MemoryType.Uploadable},
        initial_values={instance_buffer: instance_transforms.tobytes()},
    )

    shader_set = app.new_shader_set(
        "tests/instanced_cube.vert.glsl", "tests/cube.frag.glsl"
    )

    float3_format = Format.R32G32B32_FLOAT
    float4_format = Format.R32G32B32A32_FLOAT
    pipeline = app.new_pipeline(
        GraphicsPipelineDescription(
            pipeline_layout=app.new_pipeline_layout(),
            render_pass=cube_resources["render_pass"],
            vertex_shader=shader_set.instanced_cube_vert,
            fragment_shader=shader_set.cube_frag,
            vertex_attributes=[
                VertexAttribute(location=0, binding=0, format=float3_format),
                VertexAttribute(location=1, binding=1, format=float3_format),
                VertexAttribute(location=2, binding=2, format=float4_format, offset=0),
                VertexAttribute(location=3, binding=2, format=float4_format, offset=16),
                VertexAttribute(location=4, binding=2, format=float4_format, offset=32),
            ],
            vertex_bindings=[
                VertexBinding(binding=0, stride=12),
                VertexBinding(binding=1, stride=12),
                VertexBinding(
                    binding=2, stride=48, input_rate=VertexInputRate.PER_INSTANCE
                ),
            ],
            depth_description=DepthDescription(test_enabled=True, write_enabled=True),
            width=test_app.width,
            height=test_app.height,
        )
    )

    with graphics.CommandBufferBuilder(
        test_app.command_buffer, CommandBufferUsage.ONE_TIME_SUBMIT
    ) as command_buffer_builder:
        command_buffer_builder.begin_render_pass(
            render_pass=cube_resources["render_pass"],
            framebuffer=cube_resources["framebuffer"],
            width=test_app.width,
            height=test_app.height,
            clear_values=[ClearValue(color=(0.5, 0.5, 0.5, 1.0)), ClearValue(depth=1)],
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
            buffer=test_app.readback_buffer,
            width=test_app.width,
            height=test_app.height,
        )


def test_cube(test_cube_app):
    test_data, app, cube_resources = test_cube_app
    shader_set = app.new_shader_set("tests/cube.vert.glsl", "tests/cube.frag.glsl")

    pipeline = app.new_pipeline(
        GraphicsPipelineDescription(
            pipeline_layout=app.new_pipeline_layout(),
            render_pass=cube_resources["render_pass"],
            vertex_shader=shader_set.cube_vert,
            fragment_shader=shader_set.cube_frag,
            vertex_attributes=[
                VertexAttribute(location=0, binding=0, format=Format.R32G32B32_FLOAT),
                VertexAttribute(location=1, binding=1, format=Format.R32G32B32_FLOAT),
            ],
            vertex_bindings=[
                VertexBinding(binding=0, stride=12),
                VertexBinding(binding=1, stride=12),
            ],
            width=test_data.width,
            height=test_data.height,
            depth_description=DepthDescription(test_enabled=False, write_enabled=False),
        )
    )

    with graphics.CommandBufferBuilder(
        test_data.command_buffer, CommandBufferUsage.ONE_TIME_SUBMIT
    ) as command_buffer_builder:
        command_buffer_builder.begin_render_pass(
            render_pass=cube_resources["render_pass"],
            framebuffer=cube_resources["framebuffer"],
            width=test_data.width,
            height=test_data.height,
            clear_values=[ClearValue(color=(0.5, 0.5, 0.5, 1.0)), ClearValue(depth=1)],
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
