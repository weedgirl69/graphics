import subprocess
import shutil
import os

GLSLC_PATH = os.path.normpath(os.environ["GLSLC_PATH"])


def get_version_string():
    process = subprocess.run(args=[GLSLC_PATH, "--version"], stdout=subprocess.PIPE)
    return process.stdout.decode("utf-8")


def compile_shader(*, path: str, stage: str) -> bytes:
    process = subprocess.run(
        args=[GLSLC_PATH, "-fshader-stage=" + stage, "-Os", "-o", "-", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.decode("utf-8"))
    return process.stdout
