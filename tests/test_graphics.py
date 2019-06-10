import graphics
import png


def test_clear():
    WIDTH = 69
    HEIGHT = 69
    CLEAR_COLOR = [5 / 255, 144 / 255, 51 / 255, 1.0]
    GOLDEN_IMAGE_PATH = "tests/goldens/clear.png"
    with graphics.App() as app:

        image = app.new_render_target(WIDTH, HEIGHT)
        buffer = app.new_readback_buffer(WIDTH * HEIGHT * 4)
        mapped_memory = (
            app.memory_builder()
            .add(image, graphics.MemoryType.DeviceOptimal)
            .add(buffer, graphics.MemoryType.Downloadable)
            .build()
        )
        command_pool = app.new_command_pool()
        command_buffer = app.allocate_command_buffer(command_pool)

        with command_buffer.build(
            graphics.CommandBufferBuilder.ONE_TIME_SUBMIT
        ) as command_buffer_builder:
            command_buffer_builder.pipeline_barrier(
                image=image, new_layout=graphics.ImageLayout.TRANSFER_DESTINATION
            )

            command_buffer_builder.clear_color_image(image, CLEAR_COLOR)

            command_buffer_builder.pipeline_barrier(
                image=image,
                old_layout=graphics.ImageLayout.TRANSFER_DESTINATION,
                new_layout=graphics.ImageLayout.TRANSFER_SOURCE,
            )

            command_buffer_builder.copy_image_to_buffer(
                image, buffer, width=WIDTH, height=HEIGHT
            )

        app.graphics_queue.submit(command_buffer)
        app.graphics_queue.wait()

        test_image_bytes = mapped_memory[buffer][: WIDTH * HEIGHT * 4]

        try:
            with open(GOLDEN_IMAGE_PATH, "rb") as file:
                png_reader = png.Reader(file=file)
                golden_image_bytes = png_reader.read_flat()[2].tobytes()
                assert golden_image_bytes == test_image_bytes
        except FileNotFoundError:
            with open(GOLDEN_IMAGE_PATH, "wb") as file:
                png_writer = png.Writer(WIDTH, HEIGHT, alpha=True)
                png_writer.write_array(file, test_image_bytes)
