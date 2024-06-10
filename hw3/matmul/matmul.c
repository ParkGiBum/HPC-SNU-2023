#define _GNU_SOURCE
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h> 

// Naive CPU matrix multiplication
void matmul_singlethread(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float c = 0.0;
      for (size_t k = 0; k < K; k++) { c += A[i * K + k] * B[k * N + j]; }
      C[i * N + j] = c;
    }
  }
}

// void matmul_multithread(float *A, float *B, float *C, size_t M, size_t N, size_t K, int num_threads) {
//     omp_set_num_threads(num_threads);

//     #pragma omp parallel for collapse(2)
//     for (size_t i = 0; i < M; i++) {
//         for (size_t j = 0; j < N; j++) {
//             __m256 sum_vec = _mm256_setzero_ps();
//             size_t k = 0;
//             // Process the elements in chunks of 8
//             for (; k <= K - 8; k += 8) {
//                 __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
//                 // Manually multiply and accumulate for the B matrix
//                 for (int bi = 0; bi < 8; ++bi) {
//                     __m256 b_val = _mm256_broadcast_ss(&B[(k + bi) * N + j]);
//                     __m256 a_elem = _mm256_set1_ps(A[i * K + k + bi]);
//                     sum_vec = _mm256_fmadd_ps(a_elem, b_val, sum_vec);
//                 }
//             }
//             // Handle any remaining elements
//             float temp[8];
//             _mm256_storeu_ps(temp, sum_vec);
//             float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            
//             for (; k < K; ++k) {
//                 sum += A[i * K + k] * B[k * N + j];
//             }
            
//             C[i * N + j] = sum;
//         }
//     }
// }
#define TILE_SIZE 64
void matmul_multithread(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
        // Set the number of threads to use in parallel regions
    omp_set_num_threads(num_threads);
    int chunk_size = 64;
    int block_num_M = (M + TILE_SIZE - 1) / TILE_SIZE;
    int block_num_N = (N + TILE_SIZE - 1) / TILE_SIZE;
    int block_num_K = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Use OpenMP parallel region with for directive to parallelize the outer loops
    #pragma omp parallel
    {
        #pragma omp for schedule(guided, chunk_size) collapse(2)
        for (int block_row = 0; block_row < block_num_M; ++block_row) {
            for (int block_col = 0; block_col < block_num_N; ++block_col) {

                int row_start = block_row * TILE_SIZE;
                int col_start = block_col * TILE_SIZE;

                int row_end = row_start + TILE_SIZE;
                int col_end = col_start + TILE_SIZE;

                if (row_end > M) row_end = M;
                if (col_end > N) col_end = N;

                for (int block_k = 0; block_k < block_num_K; ++block_k) {
                    int k_start = block_k * TILE_SIZE;
                    int k_end = k_start + TILE_SIZE;

                    if (k_end > K) k_end = K;

                    for (int i = row_start; i < row_end; i++) {
                        int j;
                        for (j = col_start; j + 15 < col_end; j += 16) {
                            __m512 origin = _mm512_loadu_ps(&C[i * N + j]);

                            for (int k = k_start; k < k_end; k++) {
                                __m512 a = _mm512_set1_ps(A[i * K + k]);
                                __m512 b = _mm512_loadu_ps(&B[k * N + j]);
                                origin = _mm512_fmadd_ps(a, b, origin);
                            }

                            _mm512_storeu_ps(&C[i * N + j], origin);
                        }

                        // remainder 
                        for (; j < col_end; j++) {
                            float dot_product = 0.0f;
                            for (int k = k_start; k < k_end; k++) {
                                dot_product += A[i * K + k] * B[k * N + j];
                            }
                            C[i * N + j] += dot_product;
                        }
                    }
                }
            }
        }
    }
}
void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int num_threads) {
  // Naive single-threaded matmul implementation
  //matmul_singlethread(A, B, C, M, N, K);

  // TODO: Implement multi-threaded matmul
  matmul_multithread(A, B, C, M, N, K, num_threads);

}
