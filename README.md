# SparseCompress

Efficient sparse tensor compression and inference library with bit-packed storage and sliding window computation.

## Overview

SparseCompress implements a novel approach to storing and computing with sparse neural networks:
- **Compact storage**: Non-zero weights stored as a list + bit-packed position mask
- **Memory efficiency**: Up to 24x compression for 99% sparse tensors
- **Sliding window inference**: Minimize memory usage during forward pass
- **Zero reconstruction error**: Perfect numerical accuracy

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from sparse_compress import SparseCompress
import numpy as np

# Initialize compressor
compressor = SparseCompress(sparsity_threshold=1e-6)

# Create a sparse weight matrix
weight = np.random.randn(1024, 512).astype(np.float32)
weight[np.random.random((1024, 512)) < 0.9] = 0  # 90% sparsity

# Compress the weight matrix
compressed = compressor.compress(weight)
print(f"Compression ratio: {compressor.compression_ratio(weight, compressed):.2f}x")

# Perform efficient matrix multiplication with sliding window
input_batch = np.random.randn(32, 512).astype(np.float32)
output = compressor.sliding_window_matmul(compressed, input_batch, window_size=128)
```

## Key Features

### Sparse Tensor Storage
- Non-zero values stored in compact array
- Positions encoded as bit-packed mask (1 bit per element)
- Significant memory savings for sparse tensors (>80% sparsity)

### Sliding Window Inference
- Process large matrices in configurable chunks
- Reduces peak memory usage during computation
- Maintains exact numerical precision

### Performance
- **Compression**: 4-24x depending on sparsity level
- **Memory**: Up to 16x reduction with sliding window
- **Accuracy**: Zero reconstruction error

## Examples

Run the examples to see SparseCompress in action:

```bash
# Run comprehensive examples and benchmarks
python example_usage.py

# Run tests
python test_sparse_compress.py
```

## Use Cases

- Training and inference with sparse neural networks
- Memory-constrained edge deployment
- Large-scale sparse matrix operations
- Efficient storage of pruned models

## API Reference

### SparseCompress

Main class for compression and computation operations.

#### Methods

- `compress(tensor)`: Convert dense tensor to sparse format
- `decompress(sparse_tensor)`: Reconstruct full dense tensor
- `sliding_window_matmul(sparse_weight, input_tensor, window_size)`: Memory-efficient matrix multiplication
- `get_sparsity_ratio(tensor)`: Calculate fraction of zero elements
- `compression_ratio(original, compressed)`: Calculate space savings

### SparseTensor

Data structure for compressed representation.

#### Attributes

- `values`: Array of non-zero values
- `mask_bytes`: Bit-packed position mask
- `shape`: Original tensor dimensions
- `memory_size`: Total memory usage in bytes

## License

MIT