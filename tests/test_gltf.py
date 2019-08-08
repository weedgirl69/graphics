import pytest
import asyncio
import queue
import dataclasses
import math
import os
import threading
import os.path
import kt.gltf
import typing
import glob
import png
import array
import kt
import kt.command_buffer_builder
import kt.graphics_app


TEST_IMAGE_SIZE = 512
TEST_IMAGE_SAMPLE_COUNT = 3
GLTF_SAMPLE_MODELS_DIR = os.environ["GLTF_SAMPLE_MODELS_DIR"]


def cross(
    lhs: typing.Tuple[float, float, float], rhs: typing.Tuple[float, float, float]
) -> typing.Tuple[float, float, float]:
    return (
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    )


def dot(lhs: typing.Sequence[float], rhs: typing.Sequence[float]) -> float:
    return sum((x * y for x, y in zip(lhs, rhs)))


def normalize(vector: typing.Sequence[float]) -> typing.Sequence[float]:
    magnitude = math.sqrt(dot(vector, vector))
    return tuple(_ / magnitude for _ in vector)


def matrix_multiply(lhs: typing.Tuple, rhs: typing.Tuple) -> typing.Tuple:
    rows = [lhs[row_index::4] for row_index in range(4)]
    columns = [rhs[column_index * 4 :][:4] for column_index in range(4)]
    return tuple(dot(row, column) for column in columns for row in rows)


def apply_transform_sequence(
    *,
    node_transforms: typing.List[kt.gltf.AffineTransform],
    transform_sequence: typing.List[typing.Tuple[int, int]],
) -> None:
    for source_index, destination_index in transform_sequence:
        source_transform = node_transforms[source_index]
        destination_transform = node_transforms[destination_index]

        columns = [source_transform[i : i + 4] for i in range(0, 12, 4)]
        rows = [
            list(destination_transform[row_index::4]) + [row_index == 3]
            for row_index in range(4)
        ]

        node_transforms[destination_index] = tuple(
            dot(columns[i], rows[j]) for i in range(3) for j in range(4)
        )


def test_bounds() -> None:
    gltf_path = os.path.join(GLTF_SAMPLE_MODELS_DIR, "2.0/Box/glTF/Box.gltf")

    with open(gltf_path) as gltf_file:

        def read_file_bytes(uri: str) -> bytes:
            with open(os.path.join(os.path.dirname(gltf_path), uri), "rb") as file:
                return file.read()

        model = kt.gltf.from_json(gltf_file, read_file_bytes)
        scene = model.scenes[0]

        node_transforms = [
            model.node_transforms[flattened_index]
            for flattened_index in scene.node_index_to_flattened_index
        ]

        apply_transform_sequence(
            node_transforms=node_transforms,
            transform_sequence=scene.transform_source_index_to_destination_index,
        )

        bounds_center, radius = model.get_bounds(
            node_transforms=node_transforms, scene_index=0
        )
        assert bounds_center == (0.0, 0.0, 0.0)
        assert radius == 0.8660254037844386


def _render_model(
    *,
    attributes_buffer: kt.Buffer,
    command_buffer_builder: kt.command_buffer_builder.CommandBufferBuilder,
    index_buffer: kt.Buffer,
    instance_buffer: kt.Buffer,
    model: kt.gltf.Model,
    pipeline_layout: kt.PipelineLayout,
):
    scene = model.scenes[0]
    for mesh_index, node_indices in enumerate(scene.mesh_index_to_node_indices):
        instance_count = len(node_indices)
        if not instance_count:
            continue

        for primitive in model.meshes[mesh_index]:
            command_buffer_builder.push_constants(
                pipeline_layout=pipeline_layout,
                stage=kt.ShaderStage.FRAGMENT,
                byte_offset=0,
                values=int(primitive.material_index).to_bytes(4, "little"),
            )

            command_buffer_builder.bind_vertex_buffers(
                [attributes_buffer, attributes_buffer, instance_buffer],
                byte_offsets=[
                    primitive.positions_byte_offset,
                    primitive.normals_byte_offset or 0,
                    scene.mesh_index_to_base_instance_offset[mesh_index] * 3 * 4 * 4,
                ],
            )

            if primitive.index_data:

                command_buffer_builder.bind_index_buffer(
                    buffer=index_buffer,
                    index_type={2: kt.IndexType.UINT16, 4: kt.IndexType.UINT32}[
                        primitive.index_data.index_size
                    ],
                    byte_offset=primitive.index_data.byte_offset,
                )
                command_buffer_builder.draw_indexed(
                    index_count=primitive.count, instance_count=instance_count
                )
            else:
                command_buffer_builder.draw(
                    vertex_count=primitive.count, instance_count=instance_count
                )


class GltfRenderResources:
    def __init__(
        self,
        *,
        app: kt.graphics_app.GraphicsApp,
        width: int,
        height: int,
        sample_count: int,
        render_pass: kt.RenderPass,
    ):
        self.shader_set = app.new_shader_set(
            "tests/shaders/gltf.vert.glsl", "tests/shaders/gltf.frag.glsl"
        )

        self.descriptor_set_layout = app.new_descriptor_set_layout(
            [
                kt.DescriptorSetLayoutBinding(
                    stage=kt.ShaderStage.VERTEX,
                    descriptor_type=kt.DescriptorType.UNIFORM_BUFFER,
                ),
                kt.DescriptorSetLayoutBinding(
                    stage=kt.ShaderStage.FRAGMENT,
                    descriptor_type=kt.DescriptorType.UNIFORM_BUFFER,
                ),
            ]
        )

        self.pipeline_layout = app.new_pipeline_layout(
            descriptor_set_layouts=[self.descriptor_set_layout],
            push_constant_ranges=[
                kt.PushConstantRange(
                    stage=kt.ShaderStage.FRAGMENT, byte_offset=0, byte_count=4
                )
            ],
        )

        self.pipeline = app.new_graphics_pipelines(
            [
                kt.GraphicsPipelineDescription(
                    pipeline_layout=self.pipeline_layout,
                    render_pass=render_pass,
                    vertex_shader=self.shader_set.gltf_vert,
                    fragment_shader=self.shader_set.gltf_frag,
                    vertex_attributes=[
                        kt.VertexAttribute(
                            binding=0, pixel_format=kt.Format.R32G32B32_FLOAT
                        ),
                        kt.VertexAttribute(
                            binding=1, pixel_format=kt.Format.R32G32B32_FLOAT
                        ),
                    ]
                    + [
                        kt.VertexAttribute(
                            binding=2,
                            offset=i * 4 * 4,
                            pixel_format=kt.Format.R32G32B32A32_FLOAT,
                        )
                        for i in range(3)
                    ],
                    vertex_bindings=[
                        kt.VertexBinding(stride=12),
                        kt.VertexBinding(stride=12),
                        kt.VertexBinding(
                            stride=3 * 4 * 4, input_rate=kt.VertexInputRate.PER_INSTANCE
                        ),
                    ],
                    sample_count=sample_count,
                    depth_description=kt.DepthDescription(
                        test_enabled=True, write_enabled=True
                    ),
                    width=width * 2,
                    height=height * 2,
                )
            ]
        )[0]


class RendererResources:
    def __init__(
        self,
        *,
        app: kt.graphics_app.GraphicsApp,
        width: int,
        height: int,
        sample_count: int,
    ):
        self.color_target_image: kt.Image = app.new_image(
            format=kt.Format.R8G8B8A8_SRGB,
            usage=kt.ImageUsage.COLOR_ATTACHMENT,
            width=width * 2,
            height=height * 2,
            sample_count=sample_count,
        )
        self.depth_target_image: kt.Image = app.new_image(
            format=kt.Format.D24X8,
            usage=kt.ImageUsage.DEPTH_ATTACHMENT,
            width=width * 2,
            height=height * 2,
            sample_count=sample_count,
        )
        self.resolve_target_image: kt.Image = app.new_image(
            format=kt.Format.R8G8B8A8_SRGB,
            usage=kt.ImageUsage.COLOR_ATTACHMENT | kt.ImageUsage.TRANSFER_SOURCE,
            width=width * 2,
            height=height * 2,
        )
        self.downsampled_target_image: kt.Image = app.new_image(
            format=kt.Format.R8G8B8A8_SRGB,
            usage=kt.ImageUsage.TRANSFER_DESTINATION | kt.ImageUsage.TRANSFER_SOURCE,
            width=width,
            height=height,
        )
        app.new_memory_set(
            device_optimal=[self.resolve_target_image, self.downsampled_target_image],
            lazily_allocated=[self.color_target_image, self.depth_target_image],
        )
        self.color_target_view = app.new_image_view(
            image=self.color_target_image,
            format=kt.Format.R8G8B8A8_SRGB,
            aspect=kt.ImageAspect.COLOR,
        )
        self.depth_target_view = app.new_image_view(
            image=self.depth_target_image,
            format=kt.Format.D24X8,
            aspect=kt.ImageAspect.DEPTH,
        )
        self.resolve_target_view = app.new_image_view(
            image=self.resolve_target_image,
            format=kt.Format.R8G8B8A8_SRGB,
            aspect=kt.ImageAspect.COLOR,
        )
        self.render_pass = app.new_render_pass(
            attachment_descriptions=[
                kt.AttachmentDescription(
                    pixel_format=kt.Format.R8G8B8A8_SRGB,
                    load_op=kt.LoadOp.CLEAR,
                    store_op=kt.StoreOp.DISCARD,
                    # final_layout=ImageLayout.TRANSFER_SOURCE,
                    sample_count=TEST_IMAGE_SAMPLE_COUNT,
                ),
                kt.AttachmentDescription(
                    pixel_format=kt.Format.D24X8,
                    load_op=kt.LoadOp.CLEAR,
                    store_op=kt.StoreOp.DISCARD,
                    sample_count=TEST_IMAGE_SAMPLE_COUNT,
                ),
                kt.AttachmentDescription(
                    pixel_format=kt.Format.R8G8B8A8_SRGB,
                    load_op=kt.LoadOp.DONT_CARE,
                    store_op=kt.StoreOp.STORE,
                    final_layout=kt.ImageLayout.TRANSFER_SOURCE,
                ),
            ],
            subpass_descriptions=[
                kt.SubpassDescription(
                    color_attachments=[(0, kt.ImageLayout.COLOR)],
                    resolve_attachments=[(2, kt.ImageLayout.TRANSFER_DESTINATION)],
                    depth_attachment_index=1,
                )
            ],
        )
        self.framebuffer = app.new_framebuffer(
            render_pass=self.render_pass,
            attachments=[
                self.color_target_view,
                self.depth_target_view,
                self.resolve_target_view,
            ],
            width=width * 2,
            height=height * 2,
            layers=1,
        )


async def _submission_proc(
    *,
    app: kt.graphics_app.GraphicsApp,
    submission_queue: queue.Queue,
    width: int,
    height: int,
):
    write_tasks = []
    while True:
        item = submission_queue.get()
        if item is None:
            break

        command_buffer, memory_set, readback_buffer_memory, gltf_path = item

        app.graphics_queue.submit(command_buffer)
        app.graphics_queue.wait()

        async def write_image():
            test_image_bytes = readback_buffer_memory[0 : width * height * 4]
            golden_image_path = os.path.normpath(
                os.path.abspath(
                    os.path.join(
                        os.path.curdir,
                        "tests/goldens",
                        os.path.relpath(gltf_path, GLTF_SAMPLE_MODELS_DIR),
                    )
                    + ".png"
                )
            )

            os.makedirs(os.path.dirname(golden_image_path), exist_ok=True)

            with open(golden_image_path, "wb") as file:
                png_writer = png.Writer(width, height, alpha=True)
                png_writer.write_array(file, test_image_bytes)

        write_tasks.append(asyncio.create_task(write_image()))
        # app.delete_memory_set(memory_set)
    await asyncio.wait(write_tasks)


async def build_gltf_command_buffer(
    *,
    app: kt.graphics_app.GraphicsApp,
    gltf_path: str,
    gltf_render_resources: GltfRenderResources,
    width: int,
    height: int,
    renderer_resources: RendererResources,
    submission_queue: queue.Queue,
    command_pool: kt.CommandPool,
    readback_buffer_memory: kt.vk.ffi.buffer,
    readback_buffer: kt.Buffer,
):
    print(gltf_path)
    with open(gltf_path) as gltf_file:

        def read_file_bytes(uri: str):
            with open(os.path.join(os.path.dirname(gltf_path), uri), "rb") as file:
                return file.read()

        model = kt.gltf.from_json(gltf_file, read_file_bytes)
        scene = model.scenes[0]

        materials_bytes = array.array(
            "f",
            [1, 1, 1, 1]
            + [
                component
                for material in model.materials
                for component in material.base_color_factor
            ],
        ).tobytes()
        upload_buffer_byte_count = (
            len(model.indices_bytes)
            + len(model.attributes_bytes)
            + len(materials_bytes)
        )
        upload_buffer = app.new_buffer(
            byte_count=upload_buffer_byte_count, usage=kt.BufferUsage.TRANSFER_SOURCE
        )
        materials_buffer = app.new_buffer(
            byte_count=len(materials_bytes),
            usage=kt.BufferUsage.UNIFORM | kt.BufferUsage.TRANSFER_DESTINATION,
        )
        index_buffer = (
            app.new_buffer(
                byte_count=len(model.indices_bytes),
                usage=kt.BufferUsage.INDEX | kt.BufferUsage.TRANSFER_DESTINATION,
            )
            if model.indices_bytes
            else None
        )
        instance_capacity = max(
            sum(
                len(node_indices)
                for node_indices in scene.mesh_index_to_node_indices
                if node_indices
            )
            for scene in model.scenes
        )
        instance_buffer = app.new_buffer(
            byte_count=instance_capacity * 3 * 4 * 4, usage=kt.BufferUsage.VERTEX
        )
        attributes_buffer = app.new_buffer(
            byte_count=len(model.attributes_bytes),
            usage=kt.BufferUsage.VERTEX | kt.BufferUsage.TRANSFER_DESTINATION,
        )

        frame_uniform_byte_count = 4 * 4 * 4 + 4 * 4
        frame_uniform_buffer = app.new_buffer(
            byte_count=frame_uniform_byte_count, usage=kt.BufferUsage.UNIFORM
        )

        memory_set = app.new_memory_set(
            device_optimal=[materials_buffer]
            + ([index_buffer] if index_buffer else [])
            + ([attributes_buffer] if attributes_buffer else []),
            uploadable=[upload_buffer, instance_buffer, frame_uniform_buffer],
            initial_values={
                upload_buffer: model.indices_bytes
                + model.attributes_bytes
                + materials_bytes
            },
        )
        instance_memory = memory_set[instance_buffer]
        frame_uniform_memory = memory_set[frame_uniform_buffer]

        descriptor_pool = app.new_descriptor_pool(
            max_set_count=1,
            descriptor_type_counts={kt.DescriptorType.UNIFORM_BUFFER: 2},
        )
        descriptor_sets = app.allocate_descriptor_sets(
            descriptor_pool=descriptor_pool,
            descriptor_set_layouts=[gltf_render_resources.descriptor_set_layout],
        )
        app.update_descriptor_sets(
            buffer_writes=[
                kt.DescriptorBufferWrites(
                    binding=0,
                    buffer_infos=[
                        kt.DescriptorBufferInfo(
                            buffer=frame_uniform_buffer,
                            byte_count=frame_uniform_byte_count,
                            byte_offset=0,
                        )
                    ],
                    descriptor_set=descriptor_sets[0],
                    descriptor_type=kt.DescriptorType.UNIFORM_BUFFER,
                ),
                kt.DescriptorBufferWrites(
                    binding=1,
                    buffer_infos=[
                        kt.DescriptorBufferInfo(
                            buffer=materials_buffer,
                            byte_count=len(materials_bytes),
                            byte_offset=0,
                        )
                    ],
                    descriptor_set=descriptor_sets[0],
                    descriptor_type=kt.DescriptorType.UNIFORM_BUFFER,
                ),
            ]
        )

        flattened_node_transforms = [None] * len(model.node_transforms)

        for (
            node_index,
            flattened_index,
        ) in scene.node_index_to_flattened_index.items():
            flattened_node_transforms[flattened_index] = model.node_transforms[
                node_index
            ]

        transform_sequence = scene.transform_source_index_to_destination_index

        apply_transform_sequence(
            node_transforms=flattened_node_transforms,
            transform_sequence=transform_sequence,
        )

        bounds_center, radius = model.get_bounds(
            node_transforms=flattened_node_transforms, scene_index=0
        )

        scene_instance_count = sum(
            len(node_indices) for node_indices in scene.mesh_index_to_node_indices
        )
        transform_bytes = array.array(
            "f",
            (
                component
                for transform in flattened_node_transforms[:scene_instance_count]
                for component in transform
            ),
        ).tobytes()
        instance_memory[: len(transform_bytes)] = transform_bytes

        far = radius * 3.01
        near = radius * 0.99
        camera_position = (
            bounds_center[0],
            bounds_center[1],
            bounds_center[2] + radius * 2,
        )
        view_direction = (0, 0, 1)
        right = normalize(cross((0, 1, 0), view_direction))
        up = cross(view_direction, right)

        projection = (
            1.5,
            0,
            0,
            0,
            0,
            -1.5,
            0,
            0,
            0,
            0,
            far / (near - far),
            (near * far) / (near - far),
            0,
            0,
            -1,
            0,
        )
        view = (
            *(*right, -dot(right, camera_position)),
            *(*up, -dot(up, camera_position)),
            *(*view_direction, -dot(view_direction, camera_position)),
            *(0, 0, 0, 1),
        )
        view_projection = matrix_multiply(view, projection)

        frame_uniform_memory[0:64] = array.array("f", view_projection).tobytes()
        frame_uniform_memory[64:76] = array.array("f", camera_position).tobytes()
        command_buffer = app.allocate_command_buffer(command_pool)
        with kt.command_buffer_builder.CommandBufferBuilder(
            command_buffer=command_buffer, usage=kt.CommandBufferUsage.ONE_TIME_SUBMIT
        ) as command_buffer_builder:
            if index_buffer:
                command_buffer_builder.copy_buffer_to_buffer(
                    source_buffer=upload_buffer,
                    destination_buffer=index_buffer,
                    byte_count=len(model.indices_bytes),
                )
            command_buffer_builder.copy_buffer_to_buffer(
                byte_count=len(model.attributes_bytes),
                destination_buffer=attributes_buffer,
                source_buffer=upload_buffer,
                source_offset=len(model.indices_bytes),
            )
            command_buffer_builder.copy_buffer_to_buffer(
                byte_count=len(materials_bytes),
                destination_buffer=materials_buffer,
                source_buffer=upload_buffer,
                source_offset=len(model.indices_bytes) + len(model.attributes_bytes),
            )

            command_buffer_builder.bind_descriptor_sets(
                pipeline_layout=gltf_render_resources.pipeline_layout,
                descriptor_sets=[descriptor_sets[0]],
            )
            command_buffer_builder.bind_pipeline(gltf_render_resources.pipeline)

            command_buffer_builder.begin_render_pass(
                render_pass=renderer_resources.render_pass,
                framebuffer=renderer_resources.framebuffer,
                clear_values=[kt.ClearColor(1.0, 0.0, 1.0, 1.0), kt.ClearDepth(1.0)],
                width=width * 2,
                height=height * 2,
            )

            _render_model(
                attributes_buffer=attributes_buffer,
                command_buffer_builder=command_buffer_builder,
                index_buffer=index_buffer,
                instance_buffer=instance_buffer,
                model=model,
                pipeline_layout=gltf_render_resources.pipeline_layout,
            )

            command_buffer_builder.end_render_pass()

            command_buffer_builder.pipeline_barrier(
                image=renderer_resources.downsampled_target_image,
                new_layout=kt.ImageLayout.TRANSFER_DESTINATION,
            )

            command_buffer_builder.blit_image(
                source_image=renderer_resources.resolve_target_image,
                source_width=width * 2,
                source_height=height * 2,
                destination_image=renderer_resources.downsampled_target_image,
                destination_width=width,
                destination_height=height,
            )

            command_buffer_builder.pipeline_barrier(
                image=renderer_resources.downsampled_target_image,
                mip_count=1,
                old_layout=kt.ImageLayout.TRANSFER_DESTINATION,
                new_layout=kt.ImageLayout.TRANSFER_SOURCE,
            )

            command_buffer_builder.copy_image_to_buffer(
                image=renderer_resources.downsampled_target_image,
                buffer=readback_buffer,
                width=width,
                height=height,
            )

        submission_queue.put(
            (command_buffer, memory_set, readback_buffer_memory, gltf_path)
        )


@pytest.mark.asyncio
async def test_gltf() -> None:
    try:
        with kt.graphics_app.run_graphics() as app:
            app: kt.graphics_app.GraphicsApp = app
            sample_count = 3
            width = 1024
            height = 1024

            renderer_resources = RendererResources(
                app=app, width=width, height=height, sample_count=sample_count
            )

            readback_buffer = app.new_buffer(
                byte_count=width * height * 4, usage=kt.BufferUsage.TRANSFER_DESTINATION
            )

            mapped_memory = app.new_memory_set(downloadable=[readback_buffer])
            readback_buffer_memory = mapped_memory[readback_buffer]

            gltf_render_resources = GltfRenderResources(
                app=app,
                width=width,
                height=height,
                render_pass=renderer_resources.render_pass,
                sample_count=sample_count,
            )

            command_pool = app.new_command_pool()

            submission_queue = queue.Queue(maxsize=1000)

            submission_task = _submission_proc(
                app=app, submission_queue=submission_queue, width=width, height=height
            )

            tasks = [
                asyncio.create_task(
                    build_gltf_command_buffer(
                        app=app,
                        gltf_path=gltf_path,
                        width=width,
                        height=height,
                        readback_buffer=readback_buffer,
                        readback_buffer_memory=readback_buffer_memory,
                        renderer_resources=renderer_resources,
                        gltf_render_resources=gltf_render_resources,
                        submission_queue=submission_queue,
                        command_pool=command_pool,
                    )
                )
                for gltf_path in glob.glob(
                    os.path.join(GLTF_SAMPLE_MODELS_DIR, "2.0/*/glTF-Embedded/*.gltf")
                )
                + glob.glob(os.path.join(GLTF_SAMPLE_MODELS_DIR, "2.0/*/glTF/*.gltf"))
            ]
            await asyncio.gather(*tasks, submission_task)

            submission_queue.put_nowait(None)
            await submission_task
    finally:
        print(app.errors)
        assert not app.errors
