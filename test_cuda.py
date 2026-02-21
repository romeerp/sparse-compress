"""
Test suite for CUDA sparse compression implementation.
"""

import numpy as np
import pytest
import time

# Check if CUDA is available
try:
    from sparse_compress_cuda import SparseCompressCUDA, SparseTensorCUDA, CUDA_AVAILABLE
    from numba import cuda
except ImportError:
    CUDA_AVAILABLE = False

from sparse_compress import SparseCompress, SparseTensor


def skip_if_no_cuda():
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")


class TestCUDACompression:
    """Test CUDA compression/decompression functionality."""
    
    def setup_method(self):
        skip_if_no_cuda()
        self.cuda_compressor = SparseCompressCUDA(sparsity_threshold=1e-6)
        self.cpu_compressor = SparseCompress(sparsity_threshold=1e-6)
    
    def test_compress_basic(self):
        """Test basic compression."""
        tensor = np.random.randn(64, 32).astype(np.float32)
        tensor[np.random.random((64, 32)) < 0.9] = 0
        
        compressed = self.cuda_compressor.compress(tensor)
        
        assert compressed.shape == tensor.shape
        assert len(compressed.values) == np.count_nonzero(tensor)
    
    def test_compress_matches_cpu(self):
        """Verify CUDA compression matches CPU implementation."""
        tensor = np.random.randn(128, 64).astype(np.float32)
        tensor[np.random.random((128, 64)) < 0.85] = 0
        
        compressed_cuda = self.cuda_compressor.compress(tensor)
        compressed_cpu = self.cpu_compressor.compress(tensor)
        
        # Values should match (order may differ due to parallel writes)
        np.testing.assert_array_almost_equal(
            np.sort(compressed_cuda.values),
            np.sort(compressed_cpu.values),
            decimal=5
        )
    
    def test_decompress_roundtrip(self):
        """Test compress -> decompress roundtrip."""
        tensor = np.random.randn(256, 128).astype(np.float32)
        tensor[np.random.random((256, 128)) < 0.95] = 0
        
        compressed = self.cuda_compressor.compress(tensor)
        decompressed = self.cuda_compressor.decompress(compressed)
        
        # Should have zero reconstruction error
        np.testing.assert_array_almost_equal(tensor, decompressed, decimal=5)
    
    def test_high_sparsity(self):
        """Test with very high sparsity (99%)."""
        tensor = np.random.randn(512, 256).astype(np.float32)
        tensor[np.random.random((512, 256)) < 0.99] = 0
        
        compressed = self.cuda_compressor.compress(tensor)
        decompressed = self.cuda_compressor.decompress(compressed)
        
        np.testing.assert_array_almost_equal(tensor, decompressed, decimal=5)
        
        # Verify high compression ratio
        ratio = self.cuda_compressor.compression_ratio(tensor, compressed)
        assert ratio > 10, f"Expected >10x compression, got {ratio:.2f}x"
    
    def test_1d_tensor(self):
        """Test with 1D tensor."""
        tensor = np.random.randn(1024).astype(np.float32)
        tensor[np.random.random(1024) < 0.9] = 0
        
        compressed = self.cuda_compressor.compress(tensor)
        decompressed = self.cuda_compressor.decompress(compressed)
        
        np.testing.assert_array_almost_equal(tensor, decompressed, decimal=5)
    
    def test_all_zeros(self):
        """Test with all-zero tensor."""
        tensor = np.zeros((64, 64), dtype=np.float32)
        
        compressed = self.cuda_compressor.compress(tensor)
        
        assert len(compressed.values) == 0 or compressed.values[0] == 0
    
    def test_no_zeros(self):
        """Test with no zeros (dense tensor)."""
        tensor = np.random.randn(32, 32).astype(np.float32) + 1.0  # Ensure no zeros
        
        compressed = self.cuda_compressor.compress(tensor)
        decompressed = self.cuda_compressor.decompress(compressed)
        
        np.testing.assert_array_almost_equal(tensor, decompressed, decimal=5)


class TestCUDAMatmul:
    """Test CUDA sparse matrix multiplication."""
    
    def setup_method(self):
        skip_if_no_cuda()
        self.cuda_compressor = SparseCompressCUDA()
        self.cpu_compressor = SparseCompress()
    
    def test_basic_matmul(self):
        """Test basic sparse matmul."""
        weight = np.random.randn(64, 32).astype(np.float32)
        weight[np.random.random((64, 32)) < 0.9] = 0
        input_data = np.random.randn(16, 32).astype(np.float32)
        
        compressed = self.cuda_compressor.compress(weight)
        compressed = self.cuda_compressor.prepare_for_matmul(compressed)
        
        output = self.cuda_compressor.sparse_matmul(compressed, input_data)
        expected = input_data @ weight.T
        
        np.testing.assert_array_almost_equal(output, expected, decimal=4)
    
    def test_matmul_matches_cpu(self):
        """Verify CUDA matmul matches CPU implementation."""
        weight = np.random.randn(128, 64).astype(np.float32)
        weight[np.random.random((128, 64)) < 0.85] = 0
        input_data = np.random.randn(32, 64).astype(np.float32)
        
        # CUDA
        compressed_cuda = self.cuda_compressor.compress(weight)
        compressed_cuda = self.cuda_compressor.prepare_for_matmul(compressed_cuda)
        output_cuda = self.cuda_compressor.sparse_matmul(compressed_cuda, input_data)
        
        # CPU
        compressed_cpu = self.cpu_compressor.compress(weight)
        output_cpu = self.cpu_compressor.sliding_window_matmul(compressed_cpu, input_data)
        
        np.testing.assert_array_almost_equal(output_cuda, output_cpu, decimal=4)
    
    def test_large_batch(self):
        """Test with larger batch size."""
        weight = np.random.randn(256, 128).astype(np.float32)
        weight[np.random.random((256, 128)) < 0.9] = 0
        input_data = np.random.randn(256, 128).astype(np.float32)
        
        compressed = self.cuda_compressor.compress(weight)
        compressed = self.cuda_compressor.prepare_for_matmul(compressed)
        
        output = self.cuda_compressor.sparse_matmul(compressed, input_data)
        expected = input_data @ weight.T
        
        np.testing.assert_array_almost_equal(output, expected, decimal=3)
    
    def test_tiled_matmul(self):
        """Test tiled matmul kernel."""
        weight = np.random.randn(128, 64).astype(np.float32)
        weight[np.random.random((128, 64)) < 0.9] = 0
        input_data = np.random.randn(64, 64).astype(np.float32)
        
        compressed = self.cuda_compressor.compress(weight)
        compressed = self.cuda_compressor.prepare_for_matmul(compressed)
        
        output = self.cuda_compressor.sparse_matmul(compressed, input_data, use_tiled=True)
        expected = input_data @ weight.T
        
        np.testing.assert_array_almost_equal(output, expected, decimal=3)
    
    def test_sliding_window_matmul(self):
        """Test sliding window matmul."""
        weight = np.random.randn(256, 128).astype(np.float32)
        weight[np.random.random((256, 128)) < 0.9] = 0
        input_data = np.random.randn(32, 128).astype(np.float32)
        
        compressed = self.cuda_compressor.compress(weight)
        
        output = self.cuda_compressor.sliding_window_matmul(
            compressed, input_data, window_size=64
        )
        expected = input_data @ weight.T
        
        np.testing.assert_array_almost_equal(output, expected, decimal=3)


class TestCUDAPerformance:
    """Performance benchmarks for CUDA implementation."""
    
    def setup_method(self):
        skip_if_no_cuda()
        self.cuda_compressor = SparseCompressCUDA()
        self.cpu_compressor = SparseCompress()
    
    def _time_fn(self, fn, warmup=3, runs=10):
        """Time a function with warmup runs."""
        for _ in range(warmup):
            fn()
        
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            fn()
            cuda.synchronize() if CUDA_AVAILABLE else None
            times.append(time.perf_counter() - start)
        
        return np.mean(times), np.std(times)
    
    def test_compression_speedup(self):
        """Benchmark compression speedup."""
        weight = np.random.randn(1024, 512).astype(np.float32)
        weight[np.random.random((1024, 512)) < 0.95] = 0
        
        cpu_time, _ = self._time_fn(lambda: self.cpu_compressor.compress(weight))
        gpu_time, _ = self._time_fn(lambda: self.cuda_compressor.compress(weight))
        
        speedup = cpu_time / gpu_time
        print(f"\nCompression speedup: {speedup:.2f}x (CPU: {cpu_time*1000:.2f}ms, GPU: {gpu_time*1000:.2f}ms)")
        
        # GPU should be faster for large tensors
        assert speedup > 0.5, f"GPU too slow: {speedup:.2f}x"
    
    def test_matmul_speedup(self):
        """Benchmark matmul speedup."""
        weight = np.random.randn(1024, 512).astype(np.float32)
        weight[np.random.random((1024, 512)) < 0.95] = 0
        input_data = np.random.randn(128, 512).astype(np.float32)
        
        compressed_cpu = self.cpu_compressor.compress(weight)
        compressed_gpu = self.cuda_compressor.compress(weight)
        compressed_gpu = self.cuda_compressor.prepare_for_matmul(compressed_gpu)
        
        cpu_time, _ = self._time_fn(
            lambda: self.cpu_compressor.sliding_window_matmul(compressed_cpu, input_data)
        )
        gpu_time, _ = self._time_fn(
            lambda: self.cuda_compressor.sparse_matmul(compressed_gpu, input_data)
        )
        
        speedup = cpu_time / gpu_time
        print(f"\nMatmul speedup: {speedup:.2f}x (CPU: {cpu_time*1000:.2f}ms, GPU: {gpu_time*1000:.2f}ms)")


def run_quick_test():
    """Quick sanity test."""
    if not CUDA_AVAILABLE:
        print("CUDA not available, skipping test")
        return
    
    print("Running quick CUDA test...")
    
    compressor = SparseCompressCUDA()
    
    # Create sparse matrix
    weight = np.random.randn(128, 64).astype(np.float32)
    weight[np.random.random((128, 64)) < 0.9] = 0
    
    # Compress
    compressed = compressor.compress(weight)
    print(f"  Compressed {weight.shape} -> {len(compressed.values)} values")
    
    # Decompress
    decompressed = compressor.decompress(compressed)
    error = np.max(np.abs(weight - decompressed))
    print(f"  Reconstruction error: {error:.2e}")
    
    # Matmul
    input_data = np.random.randn(32, 64).astype(np.float32)
    compressed = compressor.prepare_for_matmul(compressed)
    output = compressor.sparse_matmul(compressed, input_data)
    expected = input_data @ weight.T
    matmul_error = np.max(np.abs(output - expected))
    print(f"  Matmul error: {matmul_error:.2e}")
    
    assert error < 1e-5, "Reconstruction error too high"
    assert matmul_error < 1e-3, "Matmul error too high"
    
    print("  PASSED!")


if __name__ == "__main__":
    run_quick_test()
    
    print("\nRunning full test suite...")
    pytest.main([__file__, "-v", "-x"])
