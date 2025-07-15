# setup.py

from setuptools import setup
from torch.utils import cpp_extension
import torch

# Define the extension
ext_modules = [
    cpp_extension.CUDAExtension(
        name='splash_attention',
        sources=[
            'source/splash.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': [
                '-O3',
                '--expt-relaxed-constexpr',
                '--expt-extended-lambda',
                '--use_fast_math',
                '-lineinfo',
                '--ptxas-options=-v'
            ]
        },
        include_dirs=[
            # Include PyTorch headers
            torch.utils.cpp_extension.include_paths(),
        ],
    )
]

setup(
    name='splash_attention',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.12.0",
    ],
)
