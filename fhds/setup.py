import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fhds",
    version="0.0.1",
    author="5trap",
    author_email="5trap@mpi-hd.mpg.de",
    description="Stores data in a hierarchical folder structure on the filesystem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.mpi-hd.mpg.de/Pentatrap/fhds",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)