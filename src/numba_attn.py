from numba import cuda
import numpy as np
import math
import time

@cuda.jit
def matmul_kernel(A, B, C, scale):
    # Calculate the row index of the C element and grid
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # Calculate the column index of the C element and grid
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[col, k]  # B is already transposed
        C[row, col] = tmp / math.sqrt(scale)

@cuda.jit
def softmax_kernel(mat):
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < mat.shape[0]:
        # Compute max for numerical stability
        max_val = mat[row, 0]
        for i in range(1, mat.shape[1]):
            if mat[row, i] > max_val:
                max_val = mat[row, i]
        
        # Subtract max and exponentiate
        sum_val = 0.0
        for i in range(mat.shape[1]):
            mat[row, i] = math.exp(mat[row, i] - max_val)
            sum_val += mat[row, i]
        
        # Normalize
        for i in range(mat.shape[1]):
            mat[row, i] /= sum_val


def attention(Q, K, V):
    d_k = K.shape[1]
    Q_gpu = cuda.to_device(Q)
    K_gpu = cuda.to_device(K.T)  # Transpose K for matmul
    V_gpu = cuda.to_device(V)
    output = np.zeros((Q.shape[0], V.shape[1]), dtype=np.float32)

    C_gpu = cuda.device_array((Q.shape[0], K.shape[0]), dtype=np.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = int(math.ceil(C_gpu.shape[1] / threadsperblock[1]))
    blockspergrid_y = int(math.ceil(C_gpu.shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Perform matrix multiplication QK^T
    matmul_kernel[blockspergrid, threadsperblock](Q_gpu, K_gpu, C_gpu, d_k)

    # Apply softmax to the result
    threadsperblock = 128
    blockspergrid = (int(math.ceil(C_gpu.shape[0] / threadsperblock)), 1)
    softmax_kernel[blockspergrid, threadsperblock](C_gpu)

    # Multiply by V, updating the output directly on the host
    matmul_kernel[blockspergrid, threadsperblock](C_gpu, V_gpu, output, 1)  # No scaling on output as it's the final result

    return output


def main():
    # Matrix dimensions
    B = 4  # Batch size
    T = 8  # Sequence length
    D = 16  # Feature dimension (must be square rootable for simplicity)

    # Initialize random matrices for Q, K, V
    Q = np.random.rand(B, T, D).astype(np.float32)
    K = np.random.rand(B, T, D).astype(np.float32)
    V = np.random.rand(B, T, D).astype(np.float32)

    # Flatten the batch dimension into the sequence length for simplicity in this example
    Q = Q.reshape(-1, D)
    K = K.reshape(-1, D)
    V = V.reshape(-1, D)

    # Start timing
    start_time = time.time()

    # Run the attention function
    result = attention(Q, K, V)

    # End timing
    end_time = time.time()

    print(f"Result shape: {result.shape}")
    print(f"Execution time: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    main()