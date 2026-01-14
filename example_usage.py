import numpy as np
from sparse_compress import SparseCompress, SparseTensor
import time


def create_sparse_weight_matrix(rows: int, cols: int, sparsity: float = 0.9) -> np.ndarray:
    """Create a sparse weight matrix with given sparsity."""
    matrix = np.random.randn(rows, cols).astype(np.float32)
    
    mask = np.random.random((rows, cols)) < sparsity
    matrix[mask] = 0
    
    return matrix


def benchmark_compression():
    """Benchmark compression and decompression performance."""
    print("=" * 60)
    print("SparseCompress Benchmarks")
    print("=" * 60)
    
    compressor = SparseCompress(sparsity_threshold=1e-6)
    
    sizes = [(512, 512), (1024, 1024), (2048, 2048)]
    sparsity_levels = [0.8, 0.9, 0.95, 0.99]
    
    for rows, cols in sizes:
        print(f"\nMatrix size: {rows}x{cols}")
        print("-" * 40)
        
        for sparsity in sparsity_levels:
            weight_matrix = create_sparse_weight_matrix(rows, cols, sparsity)
            
            start_time = time.time()
            compressed = compressor.compress(weight_matrix)
            compress_time = time.time() - start_time
            
            start_time = time.time()
            decompressed = compressor.decompress(compressed)
            decompress_time = time.time() - start_time
            
            actual_sparsity = compressor.get_sparsity_ratio(weight_matrix)
            compression_ratio = compressor.compression_ratio(weight_matrix, compressed)
            
            reconstruction_error = np.max(np.abs(weight_matrix - decompressed))
            
            print(f"  Sparsity: {actual_sparsity:.1%}")
            print(f"    Compression ratio: {compression_ratio:.2f}x")
            print(f"    Compress time: {compress_time*1000:.2f}ms")
            print(f"    Decompress time: {decompress_time*1000:.2f}ms")
            print(f"    Max reconstruction error: {reconstruction_error:.2e}")
            
            original_mb = weight_matrix.nbytes / (1024 * 1024)
            compressed_mb = compressed.memory_size / (1024 * 1024)
            print(f"    Memory: {original_mb:.2f}MB -> {compressed_mb:.2f}MB")


def demo_sliding_window_inference():
    """Demonstrate sliding window inference for memory-efficient computation."""
    print("\n" + "=" * 60)
    print("Sliding Window Inference Demo")
    print("=" * 60)
    
    compressor = SparseCompress()
    
    out_features = 2048
    in_features = 1024
    batch_size = 32
    window_size = 128
    
    print(f"\nConfiguration:")
    print(f"  Weight matrix: {out_features}x{in_features}")
    print(f"  Batch size: {batch_size}")
    print(f"  Window size: {window_size}")
    
    weight_matrix = create_sparse_weight_matrix(out_features, in_features, sparsity=0.95)
    input_batch = np.random.randn(batch_size, in_features).astype(np.float32)
    
    print(f"\nCompressing weight matrix...")
    compressed_weight = compressor.compress(weight_matrix)
    compression_ratio = compressor.compression_ratio(weight_matrix, compressed_weight)
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    print(f"\nPerforming sliding window inference...")
    start_time = time.time()
    sparse_output = compressor.sliding_window_matmul(
        compressed_weight, 
        input_batch, 
        window_size=window_size
    )
    sparse_time = time.time() - start_time
    
    print(f"\nPerforming standard dense matmul...")
    start_time = time.time()
    dense_output = input_batch @ weight_matrix.T
    dense_time = time.time() - start_time
    
    max_error = np.max(np.abs(sparse_output - dense_output))
    speedup = dense_time / sparse_time
    
    print(f"\nResults:")
    print(f"  Sliding window time: {sparse_time*1000:.2f}ms")
    print(f"  Dense matmul time: {dense_time*1000:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Max error: {max_error:.2e}")
    print(f"  Output shape: {sparse_output.shape}")
    
    peak_memory_window = window_size * in_features * 4 / (1024 * 1024)
    full_memory = out_features * in_features * 4 / (1024 * 1024)
    print(f"\nMemory usage (weight matrix only):")
    print(f"  Full dense: {full_memory:.2f}MB")
    print(f"  Peak window: {peak_memory_window:.2f}MB")
    print(f"  Memory reduction: {full_memory/peak_memory_window:.1f}x")


def demo_neural_network_layer():
    """Demo using SparseCompress for a neural network layer."""
    print("\n" + "=" * 60)
    print("Neural Network Layer Demo")
    print("=" * 60)
    
    compressor = SparseCompress()
    
    class SparseLinearLayer:
        def __init__(self, in_features: int, out_features: int, sparsity: float = 0.9):
            self.in_features = in_features
            self.out_features = out_features
            
            weight = create_sparse_weight_matrix(out_features, in_features, sparsity)
            self.bias = np.random.randn(out_features).astype(np.float32) * 0.1
            
            self.compressed_weight = compressor.compress(weight)
            
            print(f"Created sparse layer: {in_features} -> {out_features}")
            print(f"  Sparsity: {compressor.get_sparsity_ratio(weight):.1%}")
            print(f"  Compression: {compressor.compression_ratio(weight, self.compressed_weight):.2f}x")
        
        def forward(self, x: np.ndarray, window_size: int = 128) -> np.ndarray:
            output = compressor.sliding_window_matmul(
                self.compressed_weight, 
                x, 
                window_size=window_size
            )
            return output + self.bias
    
    layer_configs = [
        (784, 512, 0.9),
        (512, 256, 0.95),
        (256, 128, 0.95),
        (128, 10, 0.8)
    ]
    
    print("\nBuilding sparse neural network:")
    layers = []
    for in_f, out_f, sparsity in layer_configs:
        layers.append(SparseLinearLayer(in_f, out_f, sparsity))
    
    batch_size = 64
    x = np.random.randn(batch_size, 784).astype(np.float32)
    
    print(f"\nForward pass with batch size {batch_size}:")
    for i, layer in enumerate(layers):
        print(f"  Layer {i+1}: {x.shape} -> ", end="")
        x = layer.forward(x)
        x = np.maximum(x, 0)
        print(f"{x.shape}")
    
    print(f"\nFinal output shape: {x.shape}")


if __name__ == "__main__":
    benchmark_compression()
    demo_sliding_window_inference()
    demo_neural_network_layer()