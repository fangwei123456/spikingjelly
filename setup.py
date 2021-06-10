'''
python setup.py sdist bdist_wheel
python -m twine upload dist/*
'''
import setuptools
import glob
import os

from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, BuildExtension
import sys

requirements = ["torch"]


def get_extensions():
    if CUDA_HOME is None:
        print('CUDA_HOME is None. Install Without CUDA Extension')
        return None
    else:
        print('Install With CUDA Extension')
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'spikingjelly', 'cext', 'csrc')

    if sys.platform == 'win32':
        # win32 does not support cuSparse
        ext_list = ['neuron']
    else:
        ext_list = ['gemm', 'neuron']
    
    extra_compile_args = {'cxx': ['-g'], 'nvcc': ['-use_fast_math']}
    
    extension = CUDAExtension
    define_macros = [("WITH_CUDA", None)]
    ext_modules = list([
        extension(
            '_C_' + ext_name,
            glob.glob(os.path.join(extensions_dir, ext_name, '*.cpp')) + glob.glob(os.path.join(extensions_dir, ext_name, '*.cu')),
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        ) for ext_name in ext_list])
    
    return ext_modules

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()



setup(
    install_requires=install_requires,
    name="spikingjelly",
    version="0.0.0.0.5",
    author="PKU MLG, PCL, and other contributors",
    author_email="fwei@pku.edu.cn, chyq@pku.edu.cn",
    description="A deep learning framework for SNNs built on PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fangwei123456/spikingjelly",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension
    }
)