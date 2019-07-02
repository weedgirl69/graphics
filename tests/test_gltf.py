import kt.gltf


def test_gltf():
    gltf_directory = "/usr/local/google/home/stanlo/glTF-Sample-Models/2.0/Box/glTF"
    gltf_path = gltf_directory + "/Box.gltf"
    with open(gltf_path) as gltf_file:
        kt.gltf.GltfModel(gltf_file)
