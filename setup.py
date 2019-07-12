from setuptools import setup

setup(
    name="kt",
    packages=["kt", "shaderc"],
    setup_requires=["pytest-runner"],
    install_requires=["vulkan"],
    tests_require=["pytest", "pytest-cov", "pypng", "numpy"],
)
