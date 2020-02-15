import setuptools

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SpikingFlow",
    version="0.1",
    author="PKU MLG",
    author_email="fangwei123456@pku.edu.cn",
    description="A Spiking Neural Networks simulator built on PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fangwei123456/SpikingFlow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)