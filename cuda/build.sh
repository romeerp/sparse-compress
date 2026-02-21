#!/bin/bash
# Build script for CUDA sparse compression kernels

set -e

CUDA_ARCH="${CUDA_ARCH:-sm_75}"  # Default to Turing, change as needed
NVCC_FLAGS="-O3 -use_fast_math --compiler-options=-fPIC"

echo "Building CUDA kernels for $CUDA_ARCH..."

# Compile to shared library
nvcc $NVCC_FLAGS -arch=$CUDA_ARCH -shared \
    -o libsparse_kernels.so \
    sparse_kernels.cu

echo "Built libsparse_kernels.so"

# Optional: compile with debug symbols
if [ "$1" == "debug" ]; then
    nvcc -g -G -arch=$CUDA_ARCH -shared \
        -o libsparse_kernels_debug.so \
        sparse_kernels.cu
    echo "Built libsparse_kernels_debug.so (debug)"
fi

echo "Done!"
