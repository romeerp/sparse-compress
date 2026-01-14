import numpy as np
import pytest
from sparse_compress import SparseCompress, SparseTensor


class TestSparseCompress:
    
    def setup_method(self):
        self.compressor = SparseCompress(sparsity_threshold=1e-6)
    
    def test_compress_decompress_exact(self):
        """Test exact reconstruction of sparse matrix."""
        matrix = np.array([
            [1.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
            [4.0, 5.0, 0.0]
        ], dtype=np.float32)
        
        compressed = self.compressor.compress(matrix)
        decompressed = self.compressor.decompress(compressed)
        
        np.testing.assert_array_almost_equal(matrix, decompressed)
    
    def test_compress_decompress_random(self):
        """Test compression and decompression with random sparse matrix."""
        np.random.seed(42)
        matrix = np.random.randn(100, 100).astype(np.float32)
        mask = np.random.random((100, 100)) < 0.9
        matrix[mask] = 0
        
        compressed = self.compressor.compress(matrix)
        decompressed = self.compressor.decompress(compressed)
        
        np.testing.assert_array_almost_equal(matrix, decompressed)
    
    def test_bitpacking(self):
        """Test bit packing and unpacking functionality."""
        mask = np.array([True, False, True, True, False, False, True, False,
                        True, True, False, True, False, False, False, True])
        
        packed = self.compressor._pack_bits(mask)
        unpacked = self.compressor._unpack_bits(packed, len(mask))
        
        np.testing.assert_array_equal(mask, unpacked)
    
    def test_sparsity_ratio(self):
        """Test sparsity ratio calculation."""
        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        
        ratio = self.compressor.get_sparsity_ratio(matrix)
        expected = 7/9
        assert abs(ratio - expected) < 0.01
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        matrix = np.zeros((100, 100), dtype=np.float32)
        matrix[0, 0] = 1.0
        matrix[99, 99] = 2.0
        
        compressed = self.compressor.compress(matrix)
        ratio = self.compressor.compression_ratio(matrix, compressed)
        
        assert ratio > 100
    
    def test_sliding_window_matmul(self):
        """Test sliding window matrix multiplication."""
        np.random.seed(42)
        
        weight = np.random.randn(256, 128).astype(np.float32)
        mask = np.random.random((256, 128)) < 0.8
        weight[mask] = 0
        
        input_batch = np.random.randn(32, 128).astype(np.float32)
        
        compressed_weight = self.compressor.compress(weight)
        
        sparse_output = self.compressor.sliding_window_matmul(
            compressed_weight, 
            input_batch, 
            window_size=64
        )
        
        dense_output = input_batch @ weight.T
        
        np.testing.assert_array_almost_equal(sparse_output, dense_output, decimal=5)
    
    def test_sliding_window_edge_cases(self):
        """Test sliding window with edge cases."""
        weight = np.array([
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
            [0, 5, 0]
        ], dtype=np.float32)
        
        input_batch = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ], dtype=np.float32)
        
        compressed_weight = self.compressor.compress(weight)
        
        sparse_output = self.compressor.sliding_window_matmul(
            compressed_weight,
            input_batch,
            window_size=2
        )
        
        dense_output = input_batch @ weight.T
        
        np.testing.assert_array_almost_equal(sparse_output, dense_output)
    
    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        weight = np.random.randn(10, 5).astype(np.float32)
        input_batch = np.random.randn(3, 7).astype(np.float32)
        
        compressed_weight = self.compressor.compress(weight)
        
        with pytest.raises(ValueError):
            self.compressor.sliding_window_matmul(compressed_weight, input_batch)
    
    def test_empty_tensor(self):
        """Test handling of completely sparse tensor."""
        matrix = np.zeros((50, 50), dtype=np.float32)
        
        compressed = self.compressor.compress(matrix)
        decompressed = self.compressor.decompress(compressed)
        
        np.testing.assert_array_equal(matrix, decompressed)
        assert len(compressed.values) == 0
    
    def test_dense_tensor(self):
        """Test handling of completely dense tensor."""
        matrix = np.random.randn(10, 10).astype(np.float32) + 10
        
        compressed = self.compressor.compress(matrix)
        decompressed = self.compressor.decompress(compressed)
        
        np.testing.assert_array_almost_equal(matrix, decompressed)
        assert len(compressed.values) == 100
    
    def test_different_shapes(self):
        """Test compression with various tensor shapes."""
        shapes = [
            (10,),
            (5, 20),
            (3, 4, 5),
            (2, 3, 4, 5)
        ]
        
        for shape in shapes:
            tensor = np.random.randn(*shape).astype(np.float32)
            mask = np.random.random(shape) < 0.7
            tensor[mask] = 0
            
            compressed = self.compressor.compress(tensor)
            decompressed = self.compressor.decompress(compressed)
            
            np.testing.assert_array_almost_equal(tensor, decompressed)
            assert decompressed.shape == shape


def test_memory_efficiency():
    """Test that compression actually saves memory for sparse tensors."""
    compressor = SparseCompress()
    
    size = 1000
    for sparsity in [0.8, 0.9, 0.95, 0.99]:
        matrix = np.random.randn(size, size).astype(np.float32)
        mask = np.random.random((size, size)) < sparsity
        matrix[mask] = 0
        
        compressed = compressor.compress(matrix)
        
        original_size = matrix.nbytes
        compressed_size = compressed.memory_size
        
        assert compressed_size < original_size
        
        print(f"Sparsity {sparsity:.0%}: {original_size/compressed_size:.2f}x compression")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    print("\nMemory efficiency test:")
    test_memory_efficiency()