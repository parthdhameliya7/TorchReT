from setuptools import setup, find_packages
import codecs
import os

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="torchret",
    description="",
    long_description="",
    version = '0.1.1',
    long_description_content_type="text/markdown",
    author="Parth Dhameliya",
    url="https://github.com/parthdhameliya7",
    license="Apache License 2.0",
    packages=find_packages(),
    include_package_data=True,
    platforms=["linux", "unix"],
    python_requires=">=3.7",
    install_requires=requirements,
)