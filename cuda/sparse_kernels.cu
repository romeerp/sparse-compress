#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Warp-level prefix sum using shuffle intrinsics
__device__ __forceinline__ int warp_prefix_sum(int val, int lane_id) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane_id >= offset) val += n;
    }
    return val;
}

// Count set bits in a 32-bit word (population count)
__device__ __forceinline__ int popcount(uint32_t x) {
    return __popc(x);
}

// Extract bit from packed mask
__device__ __forceinline__ bool get_bit(const uint8_t* mask, int idx) {
    return (mask[idx >> 3] >> (idx & 7)) & 1;
}

// Set bit in packed mask
__device__ __forceinline__ void set_bit(uint8_t* mask, int idx) {
    atomicOr((unsigned int*)&mask[idx >> 3], 1u << (idx & 7));
}

//==============================================================================
// COMPRESSION KERNEL
// Compresses dense tensor to sparse format with bit-packed mask
//==============================================================================
__global__ void compress_kernel(
    const float* __restrict__ dense_input,
    float* __restrict__ sparse_values,
    uint8_t* __restrict__ mask_bytes,
    int* __restrict__ value_count,
    const int total_elements,
    const float threshold
) {
    __shared__ int block_prefix_sum;
    __shared__ int block_total;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // Step 1: Determine if this element is non-zero
    int is_nonzero = 0;
    float val = 0.0f;
    
    if (tid < total_elements) {
        val = dense_input[tid];
        is_nonzero = (fabsf(val) > threshold) ? 1 : 0;
    }
    
    // Step 2: Compute prefix sum within warp
    int warp_prefix = warp_prefix_sum(is_nonzero, lane_id);
    
    // Store warp totals in shared memory
    __shared__ int warp_totals[BLOCK_SIZE / WARP_SIZE];
    if (lane_id == WARP_SIZE - 1) {
        warp_totals[warp_id] = warp_prefix;
    }
    __syncthreads();
    
    // First warp computes prefix sum of warp totals
    if (warp_id == 0 && lane_id < (BLOCK_SIZE / WARP_SIZE)) {
        int warp_val = warp_totals[lane_id];
        int prefix = warp_prefix_sum(warp_val, lane_id);
        warp_totals[lane_id] = prefix;
        
        // Last thread gets block total
        if (lane_id == (BLOCK_SIZE / WARP_SIZE) - 1) {
            block_total = prefix;
        }
    }
    __syncthreads();
    
    // Each thread computes its global prefix
    int thread_prefix = warp_prefix;
    if (warp_id > 0) {
        thread_prefix += warp_totals[warp_id - 1];
    }
    
    // First thread of block atomically adds to global counter
    if (threadIdx.x == 0) {
        block_prefix_sum = atomicAdd(value_count, block_total);
    }
    __syncthreads();
    
    // Step 3: Write non-zero values and set mask bits
    if (tid < total_elements) {
        if (is_nonzero) {
            // Global index for this non-zero value
            int global_idx = block_prefix_sum + thread_prefix - 1;
            sparse_values[global_idx] = val;
        }
        
        // Set or clear mask bit
        int byte_idx = tid >> 3;
        int bit_idx = tid & 7;
        
        // Use atomic to set bits (multiple threads may write to same byte)
        if (is_nonzero) {
            atomicOr((unsigned int*)&mask_bytes[byte_idx & ~3], 
                     ((unsigned int)1 << bit_idx) << ((byte_idx & 3) * 8));
        }
    }
}

//==============================================================================
// DECOMPRESSION KERNEL  
// Reconstructs dense tensor from sparse format
//==============================================================================
__global__ void decompress_kernel(
    const float* __restrict__ sparse_values,
    const uint8_t* __restrict__ mask_bytes,
    float* __restrict__ dense_output,
    const int* __restrict__ prefix_sums,
    const int total_elements
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < total_elements) {
        int byte_idx = tid >> 3;
        int bit_idx = tid & 7;
        
        bool is_nonzero = (mask_bytes[byte_idx] >> bit_idx) & 1;
        
        if (is_nonzero) {
            // Use precomputed prefix sum to find value index
            int value_idx = prefix_sums[tid];
            dense_output[tid] = sparse_values[value_idx];
        } else {
            dense_output[tid] = 0.0f;
        }
    }
}

// Compute prefix sums for decompression (using parallel scan)
__global__ void compute_prefix_sums_kernel(
    const uint8_t* __restrict__ mask_bytes,
    int* __restrict__ prefix_sums,
    const int total_elements
) {
    extern __shared__ int shared_data[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // Load and compute local bit value
    int is_set = 0;
    if (tid < total_elements) {
        int byte_idx = tid >> 3;
        int bit_idx = tid & 7;
        is_set = (mask_bytes[byte_idx] >> bit_idx) & 1;
    }
    
    // Warp-level exclusive prefix sum
    int warp_exclusive = warp_prefix_sum(is_set, lane_id) - is_set;
    
    // Store warp totals
    __shared__ int warp_totals[BLOCK_SIZE / WARP_SIZE];
    if (lane_id == WARP_SIZE - 1) {
        warp_totals[warp_id] = warp_exclusive + is_set;
    }
    __syncthreads();
    
    // First warp computes prefix of warp totals
    if (warp_id == 0 && lane_id < (BLOCK_SIZE / WARP_SIZE)) {
        int warp_val = warp_totals[lane_id];
        warp_totals[lane_id] = warp_prefix_sum(warp_val, lane_id) - warp_val;
    }
    __syncthreads();
    
    // Final prefix sum for each thread
    int thread_prefix = warp_exclusive;
    if (warp_id > 0) {
        thread_prefix += warp_totals[warp_id];
    }
    
    if (tid < total_elements) {
        prefix_sums[tid] = thread_prefix;
    }
}

//==============================================================================
// SPARSE MATRIX MULTIPLICATION KERNEL
// Directly multiplies sparse weight matrix with dense input
// weight_shape: (out_features, in_features), stored row-major
// input: (batch_size, in_features)
// output: (batch_size, out_features)
//==============================================================================
__global__ void sparse_matmul_kernel(
    const float* __restrict__ sparse_values,
    const uint8_t* __restrict__ mask_bytes,
    const int* __restrict__ row_value_offsets,  // Start index in sparse_values for each row
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    // Each block handles one output row, threads process batch elements
    int out_idx = blockIdx.x;
    int batch_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Get pointer to sparse values for this output row
    int value_offset = row_value_offsets[out_idx];
    int next_offset = row_value_offsets[out_idx + 1];
    int num_values = next_offset - value_offset;
    
    // Mask start for this row
    int mask_start_bit = out_idx * in_features;
    
    float sum = 0.0f;
    int val_idx = value_offset;
    
    // Iterate through input features, accumulating where mask is set
    for (int in_idx = 0; in_idx < in_features; in_idx++) {
        int global_bit = mask_start_bit + in_idx;
        int byte_idx = global_bit >> 3;
        int bit_idx = global_bit & 7;
        
        if ((mask_bytes[byte_idx] >> bit_idx) & 1) {
            float weight = sparse_values[val_idx];
            float inp = input[batch_idx * in_features + in_idx];
            sum += weight * inp;
            val_idx++;
        }
    }
    
    output[batch_idx * out_features + out_idx] = sum;
}

// Optimized sparse matmul using shared memory for input reuse
__global__ void sparse_matmul_shared_kernel(
    const float* __restrict__ sparse_values,
    const uint8_t* __restrict__ mask_bytes,
    const int* __restrict__ row_value_offsets,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    extern __shared__ float shared_input[];
    
    int out_idx = blockIdx.x;
    int batch_block = blockIdx.y;
    int tid = threadIdx.x;
    
    // Each block processes BLOCK_SIZE batch elements for one output row
    int batch_start = batch_block * BLOCK_SIZE;
    int batch_idx = batch_start + tid;
    
    // Load input row to shared memory (tiled loading)
    int value_offset = row_value_offsets[out_idx];
    int mask_start_bit = out_idx * in_features;
    
    float sum = 0.0f;
    int val_idx = value_offset;
    
    // Process input in tiles
    const int TILE_SIZE = BLOCK_SIZE;
    
    for (int tile_start = 0; tile_start < in_features; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, in_features);
        
        // Cooperatively load input tile to shared memory
        for (int i = tid; i < (tile_end - tile_start) * min(BLOCK_SIZE, batch_size - batch_start); 
             i += BLOCK_SIZE) {
            int local_in_idx = i % (tile_end - tile_start);
            int local_batch_idx = i / (tile_end - tile_start);
            int global_batch_idx = batch_start + local_batch_idx;
            
            if (global_batch_idx < batch_size) {
                shared_input[local_batch_idx * TILE_SIZE + local_in_idx] = 
                    input[global_batch_idx * in_features + tile_start + local_in_idx];
            }
        }
        __syncthreads();
        
        // Process this tile
        if (batch_idx < batch_size) {
            for (int in_idx = tile_start; in_idx < tile_end; in_idx++) {
                int global_bit = mask_start_bit + in_idx;
                int byte_idx = global_bit >> 3;
                int bit_idx = global_bit & 7;
                
                if ((mask_bytes[byte_idx] >> bit_idx) & 1) {
                    float weight = sparse_values[val_idx];
                    float inp = shared_input[tid * TILE_SIZE + (in_idx - tile_start)];
                    sum += weight * inp;
                    val_idx++;
                }
            }
        }
        __syncthreads();
    }
    
    if (batch_idx < batch_size) {
        output[batch_idx * out_features + out_idx] = sum;
    }
}

// Compute row offsets for sparse matrix (prefix sum of non-zeros per row)
__global__ void compute_row_offsets_kernel(
    const uint8_t* __restrict__ mask_bytes,
    int* __restrict__ row_offsets,
    const int out_features,
    const int in_features
) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row_idx > out_features) return;
    
    if (row_idx == 0) {
        row_offsets[0] = 0;
        return;
    }
    
    // Count non-zeros in previous rows
    int count = 0;
    for (int r = 0; r < row_idx; r++) {
        int row_start_bit = r * in_features;
        for (int c = 0; c < in_features; c++) {
            int global_bit = row_start_bit + c;
            int byte_idx = global_bit >> 3;
            int bit_idx = global_bit & 7;
            
            if ((mask_bytes[byte_idx] >> bit_idx) & 1) {
                count++;
            }
        }
    }
    
    row_offsets[row_idx] = count;
}

// Optimized row offset computation using popcount
__global__ void compute_row_offsets_fast_kernel(
    const uint32_t* __restrict__ mask_words,
    int* __restrict__ row_offsets,
    const int out_features,
    const int in_features
) {
    extern __shared__ int shared_counts[];
    
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row_idx > out_features) return;
    
    // Each row has in_features bits = in_features/32 words (rounded up)
    int words_per_row = (in_features + 31) / 32;
    int row_start_word = row_idx * words_per_row;
    
    // Count non-zeros in this row using parallel reduction
    int local_count = 0;
    for (int w = tid; w < words_per_row; w += blockDim.x) {
        uint32_t word = mask_words[row_start_word + w];
        local_count += __popc(word);
    }
    
    // Warp reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
    
    // Store warp results
    if (tid % WARP_SIZE == 0) {
        shared_counts[tid / WARP_SIZE] = local_count;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (tid < (blockDim.x / WARP_SIZE)) {
        local_count = shared_counts[tid];
        for (int offset = (blockDim.x / WARP_SIZE) / 2; offset > 0; offset >>= 1) {
            local_count += __shfl_down_sync(0xffffffff, local_count, offset);
        }
        
        if (tid == 0) {
            row_offsets[row_idx + 1] = local_count;  // This row's count goes to next offset
        }
    }
}

//==============================================================================
// SLIDING WINDOW SPARSE MATMUL
// Process output in windows to minimize peak memory usage
//==============================================================================
__global__ void sliding_window_sparse_matmul_kernel(
    const float* __restrict__ sparse_values,
    const uint8_t* __restrict__ mask_bytes,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features,
    const int window_start,
    const int window_size,
    const int window_value_offset  // Start index in sparse_values for this window
) {
    int local_out_idx = blockIdx.x;
    int out_idx = window_start + local_out_idx;
    int batch_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features || local_out_idx >= window_size) return;
    
    int mask_start_bit = out_idx * in_features;
    
    // Count non-zeros before this row within the window
    int val_idx = window_value_offset;
    for (int r = window_start; r < out_idx; r++) {
        int row_start = r * in_features;
        for (int c = 0; c < in_features; c++) {
            int bit = row_start + c;
            if ((mask_bytes[bit >> 3] >> (bit & 7)) & 1) {
                val_idx++;
            }
        }
    }
    
    float sum = 0.0f;
    
    for (int in_idx = 0; in_idx < in_features; in_idx++) {
        int global_bit = mask_start_bit + in_idx;
        int byte_idx = global_bit >> 3;
        int bit_idx = global_bit & 7;
        
        if ((mask_bytes[byte_idx] >> bit_idx) & 1) {
            sum += sparse_values[val_idx] * input[batch_idx * in_features + in_idx];
            val_idx++;
        }
    }
    
    output[batch_idx * out_features + out_idx] = sum;
}

//==============================================================================
// HOST-CALLABLE WRAPPER FUNCTIONS
//==============================================================================

extern "C" {

void cuda_compress(
    const float* dense_input,
    float* sparse_values,
    uint8_t* mask_bytes,
    int* value_count,
    int total_elements,
    float threshold,
    cudaStream_t stream
) {
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Zero out mask bytes first
    int num_mask_bytes = (total_elements + 7) / 8;
    cudaMemsetAsync(mask_bytes, 0, num_mask_bytes, stream);
    cudaMemsetAsync(value_count, 0, sizeof(int), stream);
    
    compress_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        dense_input, sparse_values, mask_bytes, value_count, total_elements, threshold
    );
}

void cuda_decompress(
    const float* sparse_values,
    const uint8_t* mask_bytes,
    float* dense_output,
    int* prefix_sums,
    int total_elements,
    cudaStream_t stream
) {
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Compute prefix sums
    compute_prefix_sums_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        mask_bytes, prefix_sums, total_elements
    );
    
    // Decompress
    decompress_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        sparse_values, mask_bytes, dense_output, prefix_sums, total_elements
    );
}

void cuda_sparse_matmul(
    const float* sparse_values,
    const uint8_t* mask_bytes,
    const int* row_offsets,
    const float* input,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    cudaStream_t stream
) {
    dim3 grid(out_features, (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    sparse_matmul_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        sparse_values, mask_bytes, row_offsets, input, output,
        batch_size, in_features, out_features
    );
}

void cuda_compute_row_offsets(
    const uint8_t* mask_bytes,
    int* row_offsets,
    int out_features,
    int in_features,
    cudaStream_t stream
) {
    int num_blocks = (out_features + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    compute_row_offsets_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        mask_bytes, row_offsets, out_features, in_features
    );
}

}  // extern "C"
