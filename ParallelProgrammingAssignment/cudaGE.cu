#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",                 \
                         cudaGetErrorString(err__), __FILE__, __LINE__);      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// Normalize row k by A[k][k].
// One block is enough because this is only O(N) work.
__global__ void normalize_pivot_row(double* A, int N, int k) {
    __shared__ double pivot;

    if (threadIdx.x == 0) {
        pivot = A[k * N + k];
    }
    __syncthreads();

    for (int col = k + static_cast<int>(threadIdx.x);
        col < N;
        col += static_cast<int>(blockDim.x)) {
        A[k * N + col] /= pivot;
    }
}

// Eliminate column k from every row below k.
// One block owns one row, so no two blocks write the same row.
__global__ void eliminate_below(double* A, int N, int k) {
    const int row = k + 1 + static_cast<int>(blockIdx.x);

    __shared__ double m_shared;

    if (threadIdx.x == 0) {
        m_shared = A[row * N + k];
    }
    __syncthreads();

    const double m = m_shared;

    for (int col = k + static_cast<int>(threadIdx.x);
        col < N;
        col += static_cast<int>(blockDim.x)) {
        A[row * N + col] -= m * A[k * N + col];
    }
}

// Device-side algorithm only.
// d_A must already contain an N x N row-major matrix in device memory.
void ge_cuda_device(double* d_A, size_t N) {
    constexpr int block_size = 256;

    for (int k = 0; k < N; ++k) {
        normalize_pivot_row <<<1, block_size >>> (d_A, N, k);
        CUDA_CHECK(cudaGetLastError());

        if (k + 1 < N) {
            eliminate_below <<<N - k - 1, block_size >>> (d_A, N, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

// Convenience host wrapper: allocates, copies, runs, copies back, frees.
void ge_cuda(std::vector<double>& A, int N) {
    if (N <= 0) {
        return;
    }

    const std::size_t n = static_cast<std::size_t>(N);

    if (A.size() != n * n) {
        throw std::invalid_argument("A must be N x N");
    }

    const std::size_t bytes = n * n * sizeof(double);

    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));

    ge_cuda_device(d_A, N);

    CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
}