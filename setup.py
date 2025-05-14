from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="seurat-cca-py",
    version="0.1.0",
    author="Sunhao",
    author_email="oahsun@outlook.com",
    description="Python implementation of Seurat's CCA algorithm for single-cell data integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/python-seurat-cca",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
) 