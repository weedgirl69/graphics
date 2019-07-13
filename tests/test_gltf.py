import os
import kt.gltf
import glob
import kt

TEST_IMAGE_SIZE = 512
TEST_IMAGE_SAMPLE_COUNT = 3


def test_gltf() -> None:
    sample_models_dir = os.environ["GLTF_SAMPLE_MODELS_DIR"]

    for gltf_path in glob.glob(
        os.path.join(sample_models_dir, "2.0/*/glTF-Embedded/*.gltf")
    ) + glob.glob(os.path.join(sample_models_dir, "2.0/*/glTF/*.gltf")):
        print(gltf_path)
        with open(gltf_path) as gltf_file:

            def read_file_bytes(uri: str):
                with open(os.path.join(os.path.dirname(gltf_path), uri), "rb") as file:
                    return file.read()

            model = kt.gltf.from_json(gltf_file, read_file_bytes)
            print(len(model.indices_bytes))

            for scene in model.scenes:
                for mesh_index in (
                    mesh_index
                    for mesh_index, node_indices in scene.mesh_index_to_node_indices.items()
                    if node_indices
                ):
                    print(model.meshes[mesh_index])
            print()

    with kt.graphics_app.run_graphics() as app:
        readback_buffer = app.new_buffer(
            byte_count=TEST_IMAGE_SIZE * TEST_IMAGE_SIZE * 4,
            usage=kt.BufferUsage.TRANSFER_DESTINATION,
        )
        mapped_memory = app.new_memory_set(downloadable=[readback_buffer])
        command_pool = app.new_command_pool()
        command_buffer = app.allocate_command_buffer(command_pool)

        render_pass = app.new_render_pass(
            attachments=[
                kt.new_attachment_description(
                    pixel_format=kt.Format.R8G8B8A8_SRGB,
                    load_op=kt.LoadOp.CLEAR,
                    store_op=kt.StoreOp.DISCARD,
                    # final_layout=ImageLayout.TRANSFER_SOURCE,
                    sample_count=TEST_IMAGE_SAMPLE_COUNT,
                ),
                kt.new_attachment_description(
                    pixel_format=kt.Format.D24X8,
                    load_op=kt.LoadOp.CLEAR,
                    store_op=kt.StoreOp.DISCARD,
                    sample_count=TEST_IMAGE_SAMPLE_COUNT,
                ),
                kt.new_attachment_description(
                    pixel_format=kt.Format.R8G8B8A8_SRGB,
                    load_op=kt.LoadOp.DONT_CARE,
                    store_op=kt.StoreOp.STORE,
                    final_layout=kt.ImageLayout.TRANSFER_SOURCE,
                ),
            ],
            subpass_descriptions=[
                kt.new_subpass_description(
                    color_attachments=[(0, kt.ImageLayout.COLOR)],
                    resolve_attachments=[(2, kt.ImageLayout.TRANSFER_DESTINATION)],
                    depth_attachment=1,
                )
            ],
        )
