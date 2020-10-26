'''
python setup.py sdist bdist_wheel
python -m twine upload dist/*
'''
import setuptools

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    install_requires=install_requires,
    name="spikingjelly",
    version="0.0.0.0.002",
    author="PKU MLG and other contributors",
    author_email="fwei@pku.edu.cn, chyq@pku.edu.cn",
    description="A Spiking Neural Networks simulator built on PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fangwei123456/spikingjelly",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)