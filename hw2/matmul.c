#define _GNU_SOURCE
#include "util.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>
#define MAX_THREADS 256

struct thread_arg {
  const float *A;
  const float *B;
  float *C;
  int M;
  int N;
  int K;
  int num_threads;
  int rank; /* id of this thread */
} args[MAX_THREADS];
static pthread_t threads[MAX_THREADS];


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

#define TILE_SIZE 64
#define M_TILE_SIZE 16
#define N_TILE_SIZE 16
#define K_TILE_SIZE 16


static void *matmul_thread(void *arg) {
    struct thread_arg *input = (struct thread_arg *)arg;

    const float *A = input->A;
    const float *B = input->B;
    float *C = input->C;
    int M = input->M;
    int N = input->N;
    int K = input->K;
    int num_threads = input->num_threads;
    int rank = input->rank;

    int block_num_M = (M + TILE_SIZE - 1) / TILE_SIZE;
    int block_num_N = (N + TILE_SIZE - 1) / TILE_SIZE;
    int block_num_K = (K + TILE_SIZE - 1) / TILE_SIZE;
    int num_blocks = block_num_M * block_num_N;

    int blocks_per_thread = num_blocks / num_threads;
    int start_block = rank * blocks_per_thread;
    int end_block = start_block + blocks_per_thread;

    if (rank == num_threads - 1) {
        end_block = num_blocks;
    }

    for (int block = start_block; block < end_block; ++block) {
        int block_row = block / block_num_N;
        int block_col = block % block_num_N;

        int row_start = block_row * TILE_SIZE;
        int col_start = block_col * TILE_SIZE;

        int row_end = row_start + TILE_SIZE;
        int col_end = col_start + TILE_SIZE;

        if (row_start + TILE_SIZE > M) {
            row_end = M;
        } 
        if (col_start + TILE_SIZE > N) {
            col_end = N;
        } 

        for (int block_k = 0; block_k < block_num_K; ++block_k) {
            int k_start = block_k * TILE_SIZE;
            // divide by K
            int k_end;

            if (k_start + TILE_SIZE > K) {
                k_end = K;
            } else {
                k_end = k_start + TILE_SIZE;
            }

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
                if (j < col_end) {
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

    return NULL;
}


void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int num_threads) {

  // Naive single-threaded matmul implementation
  //matmul_singlethread(A, B, C, M, N, K);

  /*
   * TODO: Complete multi-threaded matrix multiplication and remove the matmul_singlethread call
   */ 

  if (num_threads > 256) {
    fprintf(stderr, "num_threads must be <= 256\n");
    exit(EXIT_FAILURE);
  }

  // Spawn num_thread CPU threads
  int err;
  for (int t = 0; t < num_threads; ++t) {
    args[t].A = A, args[t].B = B, args[t].C = C, args[t].M = M, args[t].N = N,
    args[t].K = K, args[t].num_threads = num_threads, args[t].rank = t;
    err = pthread_create(&threads[t], NULL, matmul_thread, (void *)&args[t]);
    if (err) {
      fprintf(stderr, "pthread_create(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }

  // Wait for spawned threads to terminate
  for (int t = 0; t < num_threads; ++t) {
    err = pthread_join(threads[t], NULL);
    if (err) {
      fprintf(stderr, "pthread_join(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }
}
