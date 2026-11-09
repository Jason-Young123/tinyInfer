# MAX_JOBS=4 python -m pip install -e . --no-build-isolation 会自动调用该文件

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = os.path.dirname(os.path.abspath(__file__))
CUTLASS_INCLUDE = os.path.join(
    ROOT,
    "third_party/cutlass/include"
)

setup(
    name="tinyinfer",

    ext_modules=[
        CUDAExtension(
            name="tinyinfer._C",

            sources=[
                "tinyinfer/csrc/bind.cpp",
                "tinyinfer/csrc/flash_attention.cu",
                "tinyinfer/csrc/flash_decoding.cu",
                "tinyinfer/csrc/rmsnorm.cu",
                "tinyinfer/csrc/rope.cu",
                "tinyinfer/csrc/linear.cu",
            ],

            # CUTLASS Header-only library
            include_dirs=[
                CUTLASS_INCLUDE,
            ],

            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                ],

                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",
                ],
            },
        ),
    ],

    cmdclass={
        "build_ext": BuildExtension,
    },
)