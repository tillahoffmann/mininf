from setuptools import find_packages, setup

description = "mininf is a minimal library to infer the parameters of probabilistic programs and " \
    "make predictions"
long_description = f"{description}. See [the documentation](https://mininf.readthedocs.io) for " \
    "details."

setup(
    name="mininf",
    version="0.1",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
)
