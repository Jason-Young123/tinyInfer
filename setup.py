# MAX_JOBS=4 python -m pip install -e . --no-build-isolation 会自动调用该文件

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="tinyinfer",
    ext_modules=[
        CUDAExtension(
            name="tinyinfer._C",
            sources=[
                "tinyinfer/csrc/flash_attention_bind.cpp", # 注意这里需要用相对路径
                "tinyinfer/csrc/flash_attention.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)


