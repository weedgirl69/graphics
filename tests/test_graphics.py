import png

import graphics
from graphics.types import (
    CommandBufferUsage,
    Format,
    ImageLayout,
    LoadOp,
    StoreOp,
    attachment_description,
    subpass_description,
)


def test_triangle():
    width = 69
    height = 69
    golden_image_path = "tests/goldens/triangle.png"

    attachments = [
        attachment_description(
            format=Format.R8G8B8A8_SRGB,
            load_op=LoadOp.CLEAR,
            store_op=StoreOp.STORE,
            final_layout=ImageLayout.TRANSFER_SOURCE,
        )
    ]

    subpass_descriptions = [subpass_description([(0, ImageLayout.COLOR)])]

    with graphics.App() as app:
        image = app.new_render_target(Format.R8G8B8A8_SRGB, width, height)
        buffer = app.new_readback_buffer(width * height * 4)
        mapped_memory = app.new_memory_set(
            (image, graphics.MemoryType.DeviceOptimal),
            (buffer, graphics.MemoryType.Downloadable),
        )
        render_pass = app.new_render_pass(
            attachments=attachments, subpass_descriptions=subpass_descriptions
        )
        image_view = app.new_image_view(image, Format.R8G8B8A8_SRGB)
        framebuffer = app.new_framebuffer(render_pass, [image_view], width, height, 1)
        pipeline_layout = app.new_pipeline_layout()
        shader_set = app.new_shader_set(
            "tests/triangle.vert.glsl", "tests/triangle.frag.glsl"
        )
        pipeline = app.new_pipeline(
            pipeline_layout=pipeline_layout,
            render_pass=render_pass,
            vertex_shader=shader_set.triangle_vert,
            fragment_shader=shader_set.triangle_frag,
            width=width,
            height=height,
        )
        command_pool = app.new_command_pool()
        command_buffer = app.allocate_command_buffer(command_pool)

        with graphics.CommandBufferBuilder(
            command_buffer, CommandBufferUsage.ONE_TIME_SUBMIT
        ) as command_buffer_builder:
            command_buffer_builder.begin_render_pass(
                render_pass=render_pass,
                framebuffer=framebuffer,
                width=width,
                height=height,
                clear_values=[(0, 0, 0, 0)],
            )

            command_buffer_builder.bind_pipeline(pipeline)

            command_buffer_builder.draw(vertex_count=3, instance_count=1)

            command_buffer_builder.end_render_pass()

            command_buffer_builder.copy_image_to_buffer(
                image=image, buffer=buffer, width=width, height=height
            )

        app.graphics_queue.submit(command_buffer)
        app.graphics_queue.wait()

        test_image_bytes = mapped_memory[buffer][: width * height * 4]

        try:
            with open(golden_image_path, "rb") as file:
                png_reader = png.Reader(file=file)
                golden_image_bytes = png_reader.read_flat()[2].tobytes()
                assert golden_image_bytes == test_image_bytes
        except FileNotFoundError:
            with open(golden_image_path, "wb") as file:
                png_writer = png.Writer(width, height, alpha=True)
                png_writer.write_array(file, test_image_bytes)


def test_clear():
    width = 69
    height = 69
    clear_color = [5 / 255, 144 / 255, 51 / 255, 1.0]
    golden_image_path = "tests/goldens/clear.png"

    with graphics.App() as app:
        image = app.new_render_target(Format.R8G8B8A8_UNORM, width, height)
        buffer = app.new_readback_buffer(width * height * 4)
        mapped_memory = app.new_memory_set(
            (image, graphics.MemoryType.DeviceOptimal),
            (buffer, graphics.MemoryType.Downloadable),
        )
        command_pool = app.new_command_pool()
        command_buffer = app.allocate_command_buffer(command_pool)

        with graphics.CommandBufferBuilder(
            command_buffer, CommandBufferUsage.ONE_TIME_SUBMIT
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
                image=image, width=width, height=height, buffer=buffer
            )

        app.graphics_queue.submit(command_buffer)
        app.graphics_queue.wait()

        test_image_bytes = mapped_memory[buffer][: width * height * 4]

        try:
            with open(golden_image_path, "rb") as file:
                png_reader = png.Reader(file=file)
                golden_image_bytes = png_reader.read_flat()[2].tobytes()
                assert golden_image_bytes == test_image_bytes
        except FileNotFoundError:
            with open(golden_image_path, "wb") as file:
                png_writer = png.Writer(width, height, alpha=True)
                png_writer.write_array(file, test_image_bytes)
