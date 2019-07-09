import shaderc
import os
import pytest


def test_path():
    assert os.path.exists(shaderc.GLSLC_PATH)


def test_get_version():
    assert shaderc.get_version_string()


def test_compile_shader():
    compiled_bytes = shaderc.compile_shader(
        path="tests/shaders/triangle.vert.glsl", stage="vert"
    )
    assert compiled_bytes


def test_compile_shader_exception():
    with pytest.raises(RuntimeError) as compile_error:
        shaderc.compile_shader(path="tests/shaders/broken.vert.glsl", stage="vert")
    assert "syntax error" in str(compile_error.value)
