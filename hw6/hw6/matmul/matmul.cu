#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

static int mpi_rank, mpi_world_size;

#define BLOCK 32

#define TM 8
#define BK 1024
#define TN 8
// #define BN 32



__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ float As[BLOCK][BLOCK];
    __shared__ float Bs[BLOCK][BLOCK];

    // Allocate thread-local cache for results in register file
    float threadResults[TM * TN] = {0.0};
    // Register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    int threadRow = ty, threadCol = tx;
    int strideA = BLOCK, strideB = BLOCK;

    // Outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Populate the shared memory (SMEM) caches
        for (uint loadOffset = 0; loadOffset < BLOCK; loadOffset += strideA) {
            As[(threadRow + loadOffset) * BK + threadCol] = A[(by * BLOCK + threadRow + loadOffset) * K + bkIdx + threadCol];
        }
        for (uint loadOffset = 0; loadOffset < BLOCK; loadOffset += strideB) {
            Bs[(threadRow + loadOffset) * BLOCK + threadCol] = B[(bkIdx + threadRow + loadOffset) * N + bx * BLOCK + threadCol];
        }
        __syncthreads();

        // Advance block tile
        A += BK;     // Move BK columns to the right
        B += BK * N; // Move BK rows down

        // Calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load relevant As & Bs entries into registers
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BLOCK + threadCol * TN + i];
            }
            // Perform outer product on register cache, accumulate into threadResults
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // Write back results to global memory
    int c = N * BLOCK * by + BLOCK * bx;
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[c + (threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = threadResults[resIdxM * TN + resIdxN];
        }
    }
}

#define NGPU 4
int num_gpus = NGPU;
static int Mbegin[NGPU], Mend[NGPU];
static int ngpu;
static cudaStream_t streams[NGPU];
static float *A_gpu[NGPU], *B_gpu[NGPU], *C_gpu[NGPU];


void matmul(float *A, float *B, float *C, int M, int N, int K) {

  if (mpi_rank != 0) return; // FIXME

  // Async memcpy H->D on each GPU
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(A_gpu[i], &A[Mbegin[i] * K],
                               (Mend[i] - Mbegin[i]) * K * sizeof(float),
                               cudaMemcpyHostToDevice, streams[i]));
    CHECK_CUDA(cudaMemcpyAsync(B_gpu[i], B, K * N * sizeof(float),
                               cudaMemcpyHostToDevice, streams[i]));
  }

  // Run kernels asynchronously on each GPU
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    dim3 blockDim(BLOCK, BLOCK);
    dim3 gridDim((N + BLOCK - 1) / BLOCK, (Mend[i] - Mbegin[i] + BLOCK - 1) / BLOCK);
    matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(
        A_gpu[i], B_gpu[i], C_gpu[i], Mend[i] - Mbegin[i], N, K);
    CHECK_CUDA(cudaGetLastError());
  }

  // Async memcpy D->H on each GPU
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(&C[Mbegin[i] * N], C_gpu[i],
                               (Mend[i] - Mbegin[i]) * N * sizeof(float),
                               cudaMemcpyDeviceToHost, streams[i]));
  }

  // Wait for all async jobs to finish
  for (int i = 0; i < ngpu; i++) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
  }
}


void matmul_initialize(int M, int N, int K) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  CHECK_CUDA(cudaGetDeviceCount(&ngpu));

  printf("[rank %d] Number of devices: %d\n", mpi_rank, ngpu);
  cudaDeviceProp props[ngpu];

  // change from here
  for (int i = 0; i < ngpu; ++i) {
    CHECK_CUDA(cudaGetDeviceProperties(&props[i], i));
    printf("[rank %d] device %d: %s\n", mpi_rank, i, props[i].name);
  }

  for (int i = 0; i < ngpu; i++) {
    Mbegin[i] = M / ngpu * i;
    Mend[i] = M / ngpu * (i + 1);
    if (i == ngpu - 1) Mend[i] = M;
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(
        cudaMalloc(&A_gpu[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B_gpu[i], K * N * sizeof(float)));
    CHECK_CUDA(
        cudaMalloc(&C_gpu[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
  }
}


void matmul_finalize() {
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(A_gpu[i]));
    CHECK_CUDA(cudaFree(B_gpu[i]));
    CHECK_CUDA(cudaFree(C_gpu[i]));
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
  }
}
