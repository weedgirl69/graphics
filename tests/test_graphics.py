import array
import pytest
import png
import graphics
from graphics.types import (
    BufferUsage,
    CommandBufferUsage,
    Format,
    GraphicsPipelineDescription,
    ImageLayout,
    IndexType,
    LoadOp,
    StoreOp,
    attachment_description,
    subpass_description,
    vertex_attribute,
    vertex_binding,
)


class _TestApp(graphics.App):
    def __init__(self, golden_path):
        super().__init__()
        self.golden_path = golden_path
        self.width = 420
        self.height = 420
        self.readback_buffer = self.new_buffer(
            byte_count=self.width * self.height * 4,
            usage=BufferUsage.TRANSFER_DESTINATION,
        )
        mapped_memory = self.new_memory_set(
            (self.readback_buffer, graphics.MemoryType.Downloadable)
        )
        self.readback_buffer_memory = mapped_memory[self.readback_buffer]
        self.command_pool = self.new_command_pool()
        self.command_buffer = self.allocate_command_buffer(self.command_pool)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.graphics_queue.submit(self.command_buffer)
        self.graphics_queue.wait()

        test_image_bytes = self.readback_buffer_memory[0 : self.width * self.height * 4]

        try:
            with open(self.golden_path, "rb") as file:
                png_reader = png.Reader(file=file)
                golden_image_bytes = png_reader.read_flat()[2].tobytes()
                assert golden_image_bytes == test_image_bytes
        except FileNotFoundError:
            with open(self.golden_path, "wb") as file:
                png_writer = png.Writer(self.width, self.height, alpha=True)
                png_writer.write_array(file, test_image_bytes)


@pytest.fixture(name="test_app")
def test_app_fixture(request):
    golden_path = "tests/goldens/" + request.function.__name__[len("test_") :] + ".png"
    with _TestApp(golden_path) as test_app:
        yield test_app


def test_cube(test_app):
    image = test_app.new_render_target(
        Format.R8G8B8A8_SRGB, test_app.width, test_app.height
    )
    index_buffer = test_app.new_buffer(byte_count=36 * 2, usage=BufferUsage.INDEX)
    positions_buffer = test_app.new_buffer(
        byte_count=6 * 4 * 3 * 4, usage=BufferUsage.VERTEX
    )
    normals_buffer = test_app.new_buffer(
        byte_count=6 * 4 * 3 * 4, usage=BufferUsage.VERTEX
    )
    mapped_memory = test_app.new_memory_set(
        (image, graphics.MemoryType.DeviceOptimal),
        (index_buffer, graphics.MemoryType.Uploadable),
        (positions_buffer, graphics.MemoryType.Uploadable),
        (normals_buffer, graphics.MemoryType.Uploadable),
    )

    quads = ((i + 0, i + 1, i + 2, i + 2, i + 3, i + 0) for i in range(24)[::4])
    mapped_memory[index_buffer][0 : 36 * 2] = array.array(
        "H", [index for quad in quads for index in quad]
    ).tobytes()
    positions = [
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
    ]
    normals = [
        (-1, 0, 0) * 4,
        (1, 0, 0) * 4,
        (0, -1, 0) * 4,
        (0, 1, 0) * 4,
        (0, 0, -1) * 4,
        (0, 0, 1) * 4,
    ]
    mapped_memory[positions_buffer][: len(positions) * 3 * 4] = array.array(
        "f", [component for position in positions for component in position]
    ).tobytes()
    mapped_memory[normals_buffer][: len(normals) * 3 * 4 * 4] = array.array(
        "f", [component for normal in normals for component in normal]
    ).tobytes()
    render_pass = test_app.new_render_pass(
        attachments=[
            attachment_description(
                format=Format.R8G8B8A8_SRGB,
                load_op=LoadOp.CLEAR,
                store_op=StoreOp.STORE,
                final_layout=ImageLayout.TRANSFER_SOURCE,
            )
        ],
        subpass_descriptions=[subpass_description([(0, ImageLayout.COLOR)])],
    )
    image_view = test_app.new_image_view(image, Format.R8G8B8A8_SRGB)
    framebuffer = test_app.new_framebuffer(
        render_pass=render_pass,
        attachments=[image_view],
        width=test_app.width,
        height=test_app.height,
        layers=1,
    )
    pipeline_layout = test_app.new_pipeline_layout()
    shader_set = test_app.new_shader_set("tests/cube.vert.glsl", "tests/cube.frag.glsl")
    pipeline_description = GraphicsPipelineDescription(
        pipeline_layout=pipeline_layout,
        render_pass=render_pass,
        vertex_shader=shader_set.cube_vert,
        fragment_shader=shader_set.cube_frag,
        vertex_attributes=[
            vertex_attribute(location=0, binding=0, format=Format.R32G32B32_FLOAT),
            vertex_attribute(location=1, binding=1, format=Format.R32G32B32_FLOAT),
        ],
        vertex_bindings=[
            vertex_binding(binding=0, stride=12),
            vertex_binding(binding=1, stride=12),
        ],
        width=test_app.width,
        height=test_app.height,
    )
    pipeline = test_app.new_pipeline(pipeline_description)

    with graphics.CommandBufferBuilder(
        test_app.command_buffer, CommandBufferUsage.ONE_TIME_SUBMIT
    ) as command_buffer_builder:
        command_buffer_builder.begin_render_pass(
            render_pass=render_pass,
            framebuffer=framebuffer,
            width=test_app.width,
            height=test_app.height,
            clear_values=[(0.5, 0.5, 0.5, 1.0)],
        )

        command_buffer_builder.bind_pipeline(pipeline)

        command_buffer_builder.bind_index_buffer(
            buffer=index_buffer, index_type=IndexType.UINT16
        )
        command_buffer_builder.bind_vertex_buffers([positions_buffer, normals_buffer])
        command_buffer_builder.draw_indexed(index_count=36)

        command_buffer_builder.end_render_pass()

        command_buffer_builder.copy_image_to_buffer(
            image=image,
            buffer=test_app.readback_buffer,
            width=test_app.width,
            height=test_app.height,
        )


def test_triangle(test_app):
    image = test_app.new_render_target(
        Format.R8G8B8A8_SRGB, test_app.width, test_app.height
    )
    test_app.new_memory_set((image, graphics.MemoryType.DeviceOptimal))
    render_pass = test_app.new_render_pass(
        attachments=[
            attachment_description(
                format=Format.R8G8B8A8_SRGB,
                load_op=LoadOp.CLEAR,
                store_op=StoreOp.STORE,
                final_layout=ImageLayout.TRANSFER_SOURCE,
            )
        ],
        subpass_descriptions=[subpass_description([(0, ImageLayout.COLOR)])],
    )
    image_view = test_app.new_image_view(image, Format.R8G8B8A8_SRGB)
    framebuffer = test_app.new_framebuffer(
        render_pass=render_pass,
        attachments=[image_view],
        width=test_app.width,
        height=test_app.height,
        layers=1,
    )
    pipeline_layout = test_app.new_pipeline_layout()
    shader_set = test_app.new_shader_set(
        "tests/triangle.vert.glsl", "tests/triangle.frag.glsl"
    )
    pipeline_description = GraphicsPipelineDescription(
        pipeline_layout=pipeline_layout,
        render_pass=render_pass,
        vertex_shader=shader_set.triangle_vert,
        fragment_shader=shader_set.triangle_frag,
        width=test_app.width,
        height=test_app.height,
    )
    pipeline = test_app.new_pipeline(pipeline_description)

    with graphics.CommandBufferBuilder(
        test_app.command_buffer, CommandBufferUsage.ONE_TIME_SUBMIT
    ) as command_buffer_builder:
        command_buffer_builder.begin_render_pass(
            render_pass=render_pass,
            framebuffer=framebuffer,
            width=test_app.width,
            height=test_app.height,
            clear_values=[(0, 0, 0, 0)],
        )

        command_buffer_builder.bind_pipeline(pipeline)

        command_buffer_builder.draw(vertex_count=3, instance_count=1)

        command_buffer_builder.end_render_pass()

        command_buffer_builder.copy_image_to_buffer(
            image=image,
            buffer=test_app.readback_buffer,
            width=test_app.width,
            height=test_app.height,
        )


def test_clear(test_app):
    clear_color = [5 / 255, 144 / 255, 51 / 255, 1.0]
    image = test_app.new_render_target(
        Format.R8G8B8A8_UNORM, test_app.width, test_app.height
    )
    test_app.new_memory_set((image, graphics.MemoryType.DeviceOptimal))

    with graphics.CommandBufferBuilder(
        test_app.command_buffer, CommandBufferUsage.ONE_TIME_SUBMIT
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
            width=test_app.width,
            height=test_app.height,
            buffer=test_app.readback_buffer,
        )

