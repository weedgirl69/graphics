import setuptools

setuptools.setup(
    name="graphics",
    packages=setuptools.find_packages(),
    install_requires=["vulkan", "pyshaderc"],
)
