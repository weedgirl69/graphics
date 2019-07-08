from setuptools import setup

setup(
    name="kt",
    packages=["kt", "shaderc"],
    install_requires=["vulkan"],
    tests_require=["pytest-cov", "pypng", "pytest"],
)
