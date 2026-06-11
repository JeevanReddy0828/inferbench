"""Ahead-of-time build for the fused bias+GELU CUDA extension.

Optional — `inferbench.kernels.cuda` JIT-compiles on first use. Use this only if
you want a prebuilt .so:

    cd inferbench/kernels/cuda && python setup.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="inferbench_fused_bias_gelu",
    ext_modules=[
        CUDAExtension(
            name="inferbench_fused_bias_gelu",
            sources=["fused_bias_gelu.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
