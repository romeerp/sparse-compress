"""
CUDA-accelerated sparse tensor compression and inference.

This module provides GPU-accelerated versions of the sparse compression operations.
Uses Numba CUDA for portable GPU acceleration without requiring external compilation.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import math

try:
    from numba import cuda, uint8, int32, float32
    import numba
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: Numba CUDA not available. Install with: pip install numba")


@dataclass
class SparseTensorCUDA:
    """GPU-resident sparse tensor representation."""
    values: np.ndarray  # Device array of non-zero values
    mask: np.ndarray    # Device array of bit-packed mask (uint8)
    shape: Tuple[int, ...]
    row_offsets: Optional[np.ndarray] = None  # For matrix operations
    
    @property
    def memory_size(self) -> int:
        return self.values.nbytes + self.mask.nbytes + 24


if CUDA_AVAILABLE:
    # =========================================================================
    # CUDA KERNEL DEFINITIONS
    # =========================================================================
    
    BLOCK_SIZE = 256
    WARP_SIZE = 32
    
    @cuda.jit
    def compress_count_kernel(dense_input, mask_out, counts, threshold):
        """
        Phase 1: Create bit mask and count non-zeros per block.
        """
        tid = cuda.grid(1)
        total = dense_input.shape[0]
        
        if tid >= total:
            return
        
        # Check if element is non-zero
        val = dense_input[tid]
        is_nonzero = abs(val) > threshold
        
        # Set bit in mask
        if is_nonzero:
            byte_idx = tid // 8
            bit_idx = tid % 8
            cuda.atomic.or_(mask_out, byte_idx, uint8(1 << bit_idx))
            # Count non-zeros per warp for efficiency
            cuda.atomic.add(counts, 0, 1)
    
    @cuda.jit
    def compress_gather_kernel(dense_input, mask, values_out, prefix_sums, threshold):
        """
        Phase 2: Gather non-zero values using prefix sums.
        """
        tid = cuda.grid(1)
        total = dense_input.shape[0]
        
        if tid >= total:
            return
        
        byte_idx = tid // 8
        bit_idx = tid % 8
        
        is_nonzero = (mask[byte_idx] >> bit_idx) & 1
        
        if is_nonzero:
            # Use exclusive prefix sum as output index
            out_idx = prefix_sums[tid]
            values_out[out_idx] = dense_input[tid]
    
    @cuda.jit
    def compute_prefix_sums_kernel(mask, prefix_sums, total_elements):
        """
        Compute exclusive prefix sums of mask bits.
        Simple sequential scan - can be optimized with parallel scan for large tensors.
        """
        tid = cuda.grid(1)
        
        if tid == 0:
            running_sum = 0
            for i in range(total_elements):
                byte_idx = i // 8
                bit_idx = i % 8
                is_set = (mask[byte_idx] >> bit_idx) & 1
                prefix_sums[i] = running_sum
                running_sum += is_set
    
    @cuda.jit
    def parallel_prefix_sum_kernel(mask, prefix_sums, block_sums, total_elements):
        """
        Block-level parallel prefix sum using shared memory.
        """
        shared = cuda.shared.array(BLOCK_SIZE, dtype=int32)
        
        tid = cuda.threadIdx.x
        gid = cuda.blockIdx.x * cuda.blockDim.x + tid
        
        # Load data
        val = 0
        if gid < total_elements:
            byte_idx = gid // 8
            bit_idx = gid % 8
            val = (mask[byte_idx] >> bit_idx) & 1
        
        shared[tid] = val
        cuda.syncthreads()
        
        # Up-sweep (reduce)
        offset = 1
        while offset < BLOCK_SIZE:
            idx = (tid + 1) * offset * 2 - 1
            if idx < BLOCK_SIZE:
                shared[idx] += shared[idx - offset]
            offset *= 2
            cuda.syncthreads()
        
        # Store block sum and clear last element
        if tid == BLOCK_SIZE - 1:
            block_sums[cuda.blockIdx.x] = shared[tid]
            shared[tid] = 0
        cuda.syncthreads()
        
        # Down-sweep
        offset = BLOCK_SIZE // 2
        while offset > 0:
            idx = (tid + 1) * offset * 2 - 1
            if idx < BLOCK_SIZE:
                temp = shared[idx - offset]
                shared[idx - offset] = shared[idx]
                shared[idx] += temp
            offset //= 2
            cuda.syncthreads()
        
        # Write result
        if gid < total_elements:
            prefix_sums[gid] = shared[tid]
    
    @cuda.jit
    def add_block_sums_kernel(prefix_sums, block_sums, total_elements):
        """
        Add block sums to get global prefix sums.
        """
        gid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        
        if gid < total_elements and cuda.blockIdx.x > 0:
            # Add sum of all previous blocks
            block_sum = 0
            for i in range(cuda.blockIdx.x):
                block_sum += block_sums[i]
            prefix_sums[gid] += block_sum
    
    @cuda.jit
    def decompress_kernel(sparse_values, mask, dense_output, prefix_sums, total_elements):
        """
        Decompress sparse tensor to dense format.
        """
        tid = cuda.grid(1)
        
        if tid >= total_elements:
            return
        
        byte_idx = tid // 8
        bit_idx = tid % 8
        
        is_nonzero = (mask[byte_idx] >> bit_idx) & 1
        
        if is_nonzero:
            value_idx = prefix_sums[tid]
            dense_output[tid] = sparse_values[value_idx]
        else:
            dense_output[tid] = 0.0
    
    @cuda.jit
    def sparse_matmul_kernel(
        sparse_values, mask, row_offsets,
        input_matrix, output_matrix,
        batch_size, in_features, out_features
    ):
        """
        Sparse matrix multiplication: output = input @ weight.T
        where weight is stored in sparse format.
        
        Each thread computes one output element.
        """
        batch_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        out_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        
        if batch_idx >= batch_size or out_idx >= out_features:
            return
        
        # Get value offset for this output row
        val_idx = row_offsets[out_idx]
        mask_start = out_idx * in_features
        
        accumulator = float32(0.0)
        
        # Iterate through input features
        for in_idx in range(in_features):
            global_bit = mask_start + in_idx
            byte_idx = global_bit // 8
            bit_idx = global_bit % 8
            
            is_nonzero = (mask[byte_idx] >> bit_idx) & 1
            
            if is_nonzero:
                weight = sparse_values[val_idx]
                inp = input_matrix[batch_idx * in_features + in_idx]
                accumulator += weight * inp
                val_idx += 1
        
        output_matrix[batch_idx * out_features + out_idx] = accumulator
    
    @cuda.jit
    def sparse_matmul_tiled_kernel(
        sparse_values, mask, row_offsets,
        input_matrix, output_matrix,
        batch_size, in_features, out_features
    ):
        """
        Tiled sparse matrix multiplication with shared memory optimization.
        Uses shared memory to cache input values for better memory coalescing.
        """
        TILE_SIZE = 32
        
        # Shared memory for input tile
        shared_input = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)
        
        batch_idx = cuda.blockIdx.y * TILE_SIZE + cuda.threadIdx.y
        out_idx = cuda.blockIdx.x
        
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        
        if out_idx >= out_features:
            return
        
        val_idx = row_offsets[out_idx]
        mask_start = out_idx * in_features
        
        accumulator = float32(0.0)
        
        # Process input in tiles
        num_tiles = (in_features + TILE_SIZE - 1) // TILE_SIZE
        
        for tile in range(num_tiles):
            tile_start = tile * TILE_SIZE
            in_idx = tile_start + tx
            
            # Load input tile cooperatively
            if batch_idx < batch_size and in_idx < in_features:
                shared_input[ty, tx] = input_matrix[batch_idx * in_features + in_idx]
            else:
                shared_input[ty, tx] = 0.0
            
            cuda.syncthreads()
            
            # Compute partial dot product for this tile
            if batch_idx < batch_size:
                for k in range(TILE_SIZE):
                    actual_in_idx = tile_start + k
                    if actual_in_idx < in_features:
                        global_bit = mask_start + actual_in_idx
                        byte_idx = global_bit // 8
                        bit_idx = global_bit % 8
                        
                        is_nonzero = (mask[byte_idx] >> bit_idx) & 1
                        
                        if is_nonzero:
                            accumulator += sparse_values[val_idx] * shared_input[ty, k]
                            val_idx += 1
            
            cuda.syncthreads()
        
        if batch_idx < batch_size:
            output_matrix[batch_idx * out_features + out_idx] = accumulator
    
    @cuda.jit
    def compute_row_offsets_kernel(mask, row_offsets, out_features, in_features):
        """
        Compute prefix sum of non-zeros per row.
        """
        row_idx = cuda.grid(1)
        
        if row_idx > out_features:
            return
        
        if row_idx == 0:
            row_offsets[0] = 0
            return
        
        # Count non-zeros in all previous rows
        count = 0
        for r in range(row_idx):
            row_start = r * in_features
            for c in range(in_features):
                bit = row_start + c
                byte_idx = bit // 8
                bit_idx = bit % 8
                if (mask[byte_idx] >> bit_idx) & 1:
                    count += 1
        
        row_offsets[row_idx] = count
    
    @cuda.jit
    def count_row_nonzeros_kernel(mask, row_counts, out_features, in_features):
        """
        Count non-zeros per row in parallel.
        """
        row_idx = cuda.grid(1)
        
        if row_idx >= out_features:
            return
        
        count = 0
        row_start = row_idx * in_features
        
        for c in range(in_features):
            bit = row_start + c
            byte_idx = bit // 8
            bit_idx = bit % 8
            if (mask[byte_idx] >> bit_idx) & 1:
                count += 1
        
        row_counts[row_idx] = count


class SparseCompressCUDA:
    """
    CUDA-accelerated sparse tensor compression and inference.
    
    Provides GPU-optimized versions of:
    - compress: Dense -> Sparse conversion
    - decompress: Sparse -> Dense reconstruction  
    - sparse_matmul: Direct sparse matrix multiplication
    """
    
    def __init__(self, sparsity_threshold: float = 1e-6, device_id: int = 0):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available. Install numba with CUDA support.")
        
        self.sparsity_threshold = sparsity_threshold
        self.device_id = device_id
        cuda.select_device(device_id)
        
        self.block_size = BLOCK_SIZE
    
    def compress(self, tensor: np.ndarray) -> SparseTensorCUDA:
        """
        Compress a dense tensor to sparse format on GPU.
        
        Args:
            tensor: Dense numpy array (will be copied to GPU)
            
        Returns:
            SparseTensorCUDA with GPU-resident data
        """
        shape = tensor.shape
        flat = tensor.flatten().astype(np.float32)
        total_elements = len(flat)
        
        # Allocate device arrays
        d_input = cuda.to_device(flat)
        num_mask_bytes = (total_elements + 7) // 8
        d_mask = cuda.to_device(np.zeros(num_mask_bytes, dtype=np.uint8))
        d_count = cuda.to_device(np.zeros(1, dtype=np.int32))
        
        # Phase 1: Create mask and count non-zeros
        threads = self.block_size
        blocks = (total_elements + threads - 1) // threads
        compress_count_kernel[blocks, threads](
            d_input, d_mask, d_count, self.sparsity_threshold
        )
        cuda.synchronize()
        
        # Get count and allocate values array
        count = d_count.copy_to_host()[0]
        d_values = cuda.device_array(max(count, 1), dtype=np.float32)
        
        # Phase 2: Compute prefix sums
        d_prefix = cuda.device_array(total_elements, dtype=np.int32)
        
        # Use simple sequential scan for now (parallel scan for large tensors)
        if total_elements > 100000:
            # Parallel block-wise prefix sum
            num_blocks = (total_elements + self.block_size - 1) // self.block_size
            d_block_sums = cuda.device_array(num_blocks, dtype=np.int32)
            
            parallel_prefix_sum_kernel[num_blocks, self.block_size](
                d_mask, d_prefix, d_block_sums, total_elements
            )
            
            add_block_sums_kernel[num_blocks, self.block_size](
                d_prefix, d_block_sums, total_elements
            )
        else:
            # Simple sequential for smaller tensors
            compute_prefix_sums_kernel[1, 1](d_mask, d_prefix, total_elements)
        
        cuda.synchronize()
        
        # Phase 3: Gather non-zero values
        compress_gather_kernel[blocks, threads](
            d_input, d_mask, d_values, d_prefix, self.sparsity_threshold
        )
        cuda.synchronize()
        
        return SparseTensorCUDA(
            values=d_values.copy_to_host(),
            mask=d_mask.copy_to_host(),
            shape=shape
        )
    
    def decompress(self, sparse: SparseTensorCUDA) -> np.ndarray:
        """
        Decompress sparse tensor to dense format.
        
        Args:
            sparse: SparseTensorCUDA to decompress
            
        Returns:
            Dense numpy array
        """
        total_elements = int(np.prod(sparse.shape))
        
        # Copy to device
        d_values = cuda.to_device(sparse.values)
        d_mask = cuda.to_device(sparse.mask)
        d_output = cuda.device_array(total_elements, dtype=np.float32)
        d_prefix = cuda.device_array(total_elements, dtype=np.int32)
        
        # Compute prefix sums
        if total_elements > 100000:
            num_blocks = (total_elements + self.block_size - 1) // self.block_size
            d_block_sums = cuda.device_array(num_blocks, dtype=np.int32)
            
            parallel_prefix_sum_kernel[num_blocks, self.block_size](
                d_mask, d_prefix, d_block_sums, total_elements
            )
            add_block_sums_kernel[num_blocks, self.block_size](
                d_prefix, d_block_sums, total_elements
            )
        else:
            compute_prefix_sums_kernel[1, 1](d_mask, d_prefix, total_elements)
        
        # Decompress
        threads = self.block_size
        blocks = (total_elements + threads - 1) // threads
        decompress_kernel[blocks, threads](
            d_values, d_mask, d_output, d_prefix, total_elements
        )
        cuda.synchronize()
        
        return d_output.copy_to_host().reshape(sparse.shape)
    
    def prepare_for_matmul(self, sparse: SparseTensorCUDA) -> SparseTensorCUDA:
        """
        Prepare sparse tensor for matrix multiplication by computing row offsets.
        """
        if len(sparse.shape) != 2:
            raise ValueError("Matrix must be 2D")
        
        out_features, in_features = sparse.shape
        
        d_mask = cuda.to_device(sparse.mask)
        d_row_offsets = cuda.device_array(out_features + 1, dtype=np.int32)
        
        threads = min(self.block_size, out_features + 1)
        blocks = (out_features + 1 + threads - 1) // threads
        
        compute_row_offsets_kernel[blocks, threads](
            d_mask, d_row_offsets, out_features, in_features
        )
        cuda.synchronize()
        
        sparse.row_offsets = d_row_offsets.copy_to_host()
        return sparse
    
    def sparse_matmul(
        self, 
        sparse_weight: SparseTensorCUDA, 
        input_tensor: np.ndarray,
        use_tiled: bool = False
    ) -> np.ndarray:
        """
        Perform sparse matrix multiplication: output = input @ weight.T
        
        Args:
            sparse_weight: Sparse weight matrix (out_features, in_features)
            input_tensor: Dense input matrix (batch_size, in_features)
            use_tiled: Use tiled kernel with shared memory (better for larger batches)
            
        Returns:
            Output matrix (batch_size, out_features)
        """
        if len(sparse_weight.shape) != 2:
            raise ValueError("Weight must be 2D")
        if len(input_tensor.shape) != 2:
            raise ValueError("Input must be 2D")
        
        out_features, in_features = sparse_weight.shape
        batch_size = input_tensor.shape[0]
        
        if input_tensor.shape[1] != in_features:
            raise ValueError(f"Input dimension mismatch: {input_tensor.shape[1]} != {in_features}")
        
        # Ensure row offsets are computed
        if sparse_weight.row_offsets is None:
            sparse_weight = self.prepare_for_matmul(sparse_weight)
        
        # Copy to device
        d_values = cuda.to_device(sparse_weight.values)
        d_mask = cuda.to_device(sparse_weight.mask)
        d_row_offsets = cuda.to_device(sparse_weight.row_offsets)
        d_input = cuda.to_device(input_tensor.flatten().astype(np.float32))
        d_output = cuda.device_array(batch_size * out_features, dtype=np.float32)
        
        if use_tiled:
            TILE_SIZE = 32
            blocks_x = out_features
            blocks_y = (batch_size + TILE_SIZE - 1) // TILE_SIZE
            
            sparse_matmul_tiled_kernel[(blocks_x, blocks_y), (TILE_SIZE, TILE_SIZE)](
                d_values, d_mask, d_row_offsets,
                d_input, d_output,
                batch_size, in_features, out_features
            )
        else:
            threads_x = min(32, out_features)
            threads_y = min(32, batch_size)
            blocks_x = (out_features + threads_x - 1) // threads_x
            blocks_y = (batch_size + threads_y - 1) // threads_y
            
            sparse_matmul_kernel[(blocks_x, blocks_y), (threads_x, threads_y)](
                d_values, d_mask, d_row_offsets,
                d_input, d_output,
                batch_size, in_features, out_features
            )
        
        cuda.synchronize()
        
        return d_output.copy_to_host().reshape(batch_size, out_features)
    
    def sliding_window_matmul(
        self,
        sparse_weight: SparseTensorCUDA,
        input_tensor: np.ndarray,
        window_size: int = 128
    ) -> np.ndarray:
        """
        Memory-efficient sparse matmul using sliding window.
        
        Processes output rows in windows to minimize peak GPU memory.
        """
        if len(sparse_weight.shape) != 2:
            raise ValueError("Weight must be 2D")
        
        out_features, in_features = sparse_weight.shape
        batch_size = input_tensor.shape[0]
        
        output = np.zeros((batch_size, out_features), dtype=np.float32)
        
        # Process in windows
        for window_start in range(0, out_features, window_size):
            window_end = min(window_start + window_size, out_features)
            actual_window_size = window_end - window_start
            
            # Extract window from sparse representation
            window_mask_start = (window_start * in_features) // 8
            window_mask_end = ((window_end * in_features) + 7) // 8
            
            # Create window sparse tensor
            window_sparse = SparseTensorCUDA(
                values=sparse_weight.values,
                mask=sparse_weight.mask,
                shape=(actual_window_size, in_features)
            )
            
            # Compute offsets for this window
            total_bits = window_start * in_features
            window_offset = 0
            for i in range(total_bits):
                byte_idx = i // 8
                bit_idx = i % 8
                if (sparse_weight.mask[byte_idx] >> bit_idx) & 1:
                    window_offset += 1
            
            # Extract window values
            window_count = 0
            for r in range(window_start, window_end):
                for c in range(in_features):
                    bit = r * in_features + c
                    if (sparse_weight.mask[bit // 8] >> (bit % 8)) & 1:
                        window_count += 1
            
            window_values = sparse_weight.values[window_offset:window_offset + window_count]
            
            # Build window mask
            window_mask_bytes = (actual_window_size * in_features + 7) // 8
            window_mask = np.zeros(window_mask_bytes, dtype=np.uint8)
            
            for r in range(actual_window_size):
                for c in range(in_features):
                    src_bit = (window_start + r) * in_features + c
                    dst_bit = r * in_features + c
                    if (sparse_weight.mask[src_bit // 8] >> (src_bit % 8)) & 1:
                        window_mask[dst_bit // 8] |= (1 << (dst_bit % 8))
            
            window_tensor = SparseTensorCUDA(
                values=window_values,
                mask=window_mask,
                shape=(actual_window_size, in_features)
            )
            
            # Compute matmul for this window
            window_output = self.sparse_matmul(window_tensor, input_tensor)
            output[:, window_start:window_end] = window_output
        
        return output
    
    def get_sparsity_ratio(self, tensor: np.ndarray) -> float:
        """Calculate sparsity ratio of a tensor."""
        total = tensor.size
        zeros = np.sum(np.abs(tensor.flatten()) <= self.sparsity_threshold)
        return zeros / total
    
    def compression_ratio(self, original: np.ndarray, compressed: SparseTensorCUDA) -> float:
        """Calculate compression ratio."""
        original_size = original.nbytes
        compressed_size = compressed.memory_size
        return original_size / compressed_size


def benchmark_cuda_vs_cpu():
    """Benchmark CUDA vs CPU implementations."""
    from sparse_compress import SparseCompress
    import time
    
    print("=" * 60)
    print("CUDA vs CPU Benchmark")
    print("=" * 60)
    
    # Test configurations
    configs = [
        (512, 256, 32, 0.9),    # Small
        (1024, 512, 64, 0.9),   # Medium
        (2048, 1024, 128, 0.95), # Large, high sparsity
    ]
    
    for out_f, in_f, batch, sparsity in configs:
        print(f"\nConfig: weight=({out_f}, {in_f}), batch={batch}, sparsity={sparsity*100}%")
        print("-" * 40)
        
        # Create test data
        weight = np.random.randn(out_f, in_f).astype(np.float32)
        weight[np.random.random((out_f, in_f)) < sparsity] = 0
        input_data = np.random.randn(batch, in_f).astype(np.float32)
        
        # CPU implementation
        cpu = SparseCompress()
        
        start = time.perf_counter()
        compressed_cpu = cpu.compress(weight)
        cpu_compress_time = time.perf_counter() - start
        
        start = time.perf_counter()
        output_cpu = cpu.sliding_window_matmul(compressed_cpu, input_data)
        cpu_matmul_time = time.perf_counter() - start
        
        if CUDA_AVAILABLE:
            # CUDA implementation
            gpu = SparseCompressCUDA()
            
            # Warmup
            _ = gpu.compress(weight)
            cuda.synchronize()
            
            start = time.perf_counter()
            compressed_gpu = gpu.compress(weight)
            cuda.synchronize()
            gpu_compress_time = time.perf_counter() - start
            
            # Warmup
            compressed_gpu = gpu.prepare_for_matmul(compressed_gpu)
            _ = gpu.sparse_matmul(compressed_gpu, input_data)
            cuda.synchronize()
            
            start = time.perf_counter()
            output_gpu = gpu.sparse_matmul(compressed_gpu, input_data)
            cuda.synchronize()
            gpu_matmul_time = time.perf_counter() - start
            
            # Verify correctness
            max_diff = np.max(np.abs(output_cpu - output_gpu))
            
            print(f"  Compress: CPU={cpu_compress_time*1000:.2f}ms, GPU={gpu_compress_time*1000:.2f}ms, Speedup={cpu_compress_time/gpu_compress_time:.1f}x")
            print(f"  Matmul:   CPU={cpu_matmul_time*1000:.2f}ms, GPU={gpu_matmul_time*1000:.2f}ms, Speedup={cpu_matmul_time/gpu_matmul_time:.1f}x")
            print(f"  Max diff: {max_diff:.2e}")
        else:
            print(f"  CPU Compress: {cpu_compress_time*1000:.2f}ms")
            print(f"  CPU Matmul:   {cpu_matmul_time*1000:.2f}ms")
            print("  (CUDA not available)")


if __name__ == "__main__":
    if not CUDA_AVAILABLE:
        print("CUDA not available. Please install numba with CUDA support:")
        print("  pip install numba")
        print("  # Also need CUDA toolkit installed")
        exit(1)
    
    print("Testing CUDA Sparse Compression")
    print("=" * 40)
    
    # Basic test
    compressor = SparseCompressCUDA()
    
    # Create test data
    weight = np.random.randn(256, 128).astype(np.float32)
    weight[np.random.random((256, 128)) < 0.9] = 0  # 90% sparse
    
    print(f"Original shape: {weight.shape}")
    print(f"Sparsity: {compressor.get_sparsity_ratio(weight)*100:.1f}%")
    
    # Compress
    compressed = compressor.compress(weight)
    print(f"Compressed: {len(compressed.values)} values, {len(compressed.mask)} mask bytes")
    print(f"Compression ratio: {compressor.compression_ratio(weight, compressed):.2f}x")
    
    # Decompress and verify
    decompressed = compressor.decompress(compressed)
    max_error = np.max(np.abs(weight - decompressed))
    print(f"Max reconstruction error: {max_error:.2e}")
    
    # Test matmul
    input_data = np.random.randn(32, 128).astype(np.float32)
    
    compressed = compressor.prepare_for_matmul(compressed)
    output = compressor.sparse_matmul(compressed, input_data)
    
    # Compare with dense matmul
    expected = input_data @ weight.T
    matmul_error = np.max(np.abs(output - expected))
    print(f"Max matmul error: {matmul_error:.2e}")
    
    print("\nRunning benchmark...")
    benchmark_cuda_vs_cpu()
