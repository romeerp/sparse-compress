import numpy as np
from typing import Tuple, List, Optional
import struct
from dataclasses import dataclass


@dataclass
class SparseTensor:
    """Represents a sparse tensor with bitpacked mask and non-zero values."""
    values: np.ndarray
    mask_bytes: bytes
    shape: Tuple[int, ...]
    
    @property
    def memory_size(self) -> int:
        """Calculate memory usage in bytes."""
        values_size = self.values.nbytes
        mask_size = len(self.mask_bytes)
        return values_size + mask_size + 24


class SparseCompress:
    """Efficient sparse tensor compression and decompression with sliding window inference."""
    
    def __init__(self, sparsity_threshold: float = 1e-6):
        """
        Initialize SparseCompress.
        
        Args:
            sparsity_threshold: Values with absolute value below this are considered zero
        """
        self.sparsity_threshold = sparsity_threshold
    
    def compress(self, tensor: np.ndarray) -> SparseTensor:
        """
        Compress a dense tensor into sparse representation.
        
        Args:
            tensor: Dense numpy array to compress
            
        Returns:
            SparseTensor with values list and bitpacked mask
        """
        flat_tensor = tensor.flatten()
        
        mask = np.abs(flat_tensor) > self.sparsity_threshold
        
        non_zero_values = flat_tensor[mask]
        
        mask_bytes = self._pack_bits(mask)
        
        return SparseTensor(
            values=non_zero_values,
            mask_bytes=mask_bytes,
            shape=tensor.shape
        )
    
    def decompress(self, sparse_tensor: SparseTensor) -> np.ndarray:
        """
        Reconstruct the full dense tensor from sparse representation.
        
        Args:
            sparse_tensor: SparseTensor to decompress
            
        Returns:
            Reconstructed dense tensor
        """
        total_elements = np.prod(sparse_tensor.shape)
        
        mask = self._unpack_bits(sparse_tensor.mask_bytes, total_elements)
        
        dense_flat = np.zeros(total_elements, dtype=sparse_tensor.values.dtype)
        dense_flat[mask] = sparse_tensor.values
        
        return dense_flat.reshape(sparse_tensor.shape)
    
    def sliding_window_matmul(self, 
                            sparse_weight: SparseTensor,
                            input_tensor: np.ndarray,
                            window_size: int = 128) -> np.ndarray:
        """
        Perform matrix multiplication using sliding window to minimize memory usage.
        
        Args:
            sparse_weight: Sparse weight matrix (2D)
            input_tensor: Input tensor (2D: batch_size x input_dim)
            window_size: Size of sliding window for processing
            
        Returns:
            Output tensor after matrix multiplication
        """
        if len(sparse_weight.shape) != 2:
            raise ValueError("Weight matrix must be 2D")
        if len(input_tensor.shape) != 2:
            raise ValueError("Input tensor must be 2D")
        
        out_features, in_features = sparse_weight.shape
        batch_size = input_tensor.shape[0]
        
        if input_tensor.shape[1] != in_features:
            raise ValueError(f"Input dimension mismatch: {input_tensor.shape[1]} != {in_features}")
        
        output = np.zeros((batch_size, out_features), dtype=input_tensor.dtype)
        
        total_elements = np.prod(sparse_weight.shape)
        mask = self._unpack_bits(sparse_weight.mask_bytes, total_elements)
        
        value_idx = 0
        
        for out_idx in range(0, out_features, window_size):
            window_end = min(out_idx + window_size, out_features)
            window_rows = window_end - out_idx
            
            window_start_flat = out_idx * in_features
            window_end_flat = window_end * in_features
            window_mask = mask[window_start_flat:window_end_flat]
            
            num_values = np.sum(window_mask)
            window_values = sparse_weight.values[value_idx:value_idx + num_values]
            value_idx += num_values
            
            window_dense = np.zeros((window_rows, in_features), dtype=sparse_weight.values.dtype)
            window_dense_flat = window_dense.flatten()
            window_dense_flat[window_mask] = window_values
            window_dense = window_dense_flat.reshape((window_rows, in_features))
            
            output[:, out_idx:window_end] = input_tensor @ window_dense.T
        
        return output
    
    def _pack_bits(self, mask: np.ndarray) -> bytes:
        """
        Pack boolean mask into bytes.
        
        Args:
            mask: Boolean numpy array
            
        Returns:
            Packed bytes representation
        """
        mask_flat = mask.flatten()
        num_bits = len(mask_flat)
        num_bytes = (num_bits + 7) // 8
        
        packed = bytearray(num_bytes)
        
        for i, bit in enumerate(mask_flat):
            if bit:
                byte_idx = i // 8
                bit_idx = i % 8
                packed[byte_idx] |= (1 << bit_idx)
        
        return bytes(packed)
    
    def _unpack_bits(self, packed: bytes, num_bits: int) -> np.ndarray:
        """
        Unpack bytes into boolean mask.
        
        Args:
            packed: Packed bytes
            num_bits: Number of bits to unpack
            
        Returns:
            Boolean numpy array
        """
        mask = np.zeros(num_bits, dtype=bool)
        
        for i in range(num_bits):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < len(packed):
                mask[i] = (packed[byte_idx] >> bit_idx) & 1
        
        return mask
    
    def get_sparsity_ratio(self, tensor: np.ndarray) -> float:
        """
        Calculate sparsity ratio of a tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Ratio of zero elements to total elements
        """
        total = tensor.size
        zeros = np.sum(np.abs(tensor.flatten()) <= self.sparsity_threshold)
        return zeros / total
    
    def compression_ratio(self, original: np.ndarray, compressed: SparseTensor) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original: Original dense tensor
            compressed: Compressed sparse tensor
            
        Returns:
            Ratio of original size to compressed size
        """
        original_size = original.nbytes
        compressed_size = compressed.memory_size
        return original_size / compressed_size