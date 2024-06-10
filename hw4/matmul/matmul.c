#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16
#define TILE_SIZE_M 128
#define TILE_SIZE_N 64
#define TILE_SIZE_K 32
void matmul_multithread(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
    omp_set_num_threads(num_threads);


    int chunk_size = TILE_SIZE;
    int block_num_M = (M + TILE_SIZE_M - 1) / TILE_SIZE_M;
    int block_num_N = (N + TILE_SIZE_N - 1) / TILE_SIZE_N;
    int block_num_K = (K + TILE_SIZE_K - 1) / TILE_SIZE_K;

    // Use OpenMP parallel region with for directive to parallelize the outer loops
    #pragma omp parallel
    {
        #pragma omp for schedule(guided, chunk_size) collapse(2)
        for (int block_row = 0; block_row < block_num_M; ++block_row) {
            for (int block_col = 0; block_col < block_num_N; ++block_col) {

                int row_start = block_row * TILE_SIZE_M;
                int col_start = block_col * TILE_SIZE_N;

                int row_end = row_start + TILE_SIZE_M;
                int col_end = col_start + TILE_SIZE_N;

                // if (row_end > M) row_end = M;
                // if (col_end > N) col_end = N;

                for (int block_k = 0; block_k < block_num_K; ++block_k) {
                    int k_start = block_k * TILE_SIZE_K;
                    int k_end = k_start + TILE_SIZE_K;

                    // if (k_end > K) k_end = K;

                    for (int i = row_start; i < row_end; i++) {
                        for (int j = col_start; j + 15 < col_end; j += 16) {
                            __m512 origin = _mm512_loadu_ps(&C[i * N + j]);

                            for (int k = k_start; k < k_end; k++) {
                                __m512 a = _mm512_set1_ps(A[i * K + k]);
                                __m512 b = _mm512_loadu_ps(&B[k * N + j]);
                                origin = _mm512_fmadd_ps(a, b, origin);
                            }

                            _mm512_storeu_ps(&C[i * N + j], origin);
                        }

                    }
                }
            }
        }
    }
}

void matmul_distributed( float *A,  float *B, float *C, int M, int N, int K, int num_threads, int mpi_rank, int mpi_size) {
    int rows_per_process = M / mpi_size;
    int remaining_rows = M % mpi_size;

    int start_row;
    int end_row;
    if (mpi_rank < remaining_rows) {
        start_row = mpi_rank * rows_per_process + mpi_rank;
        end_row = start_row + rows_per_process + 1;
    } else {
        start_row = mpi_rank * rows_per_process + remaining_rows;
        end_row = start_row + rows_per_process;
    }

    float *sub_A = A + start_row * K;
    float *sub_C = C + start_row * N;

    int *sendcounts = (int *)malloc(mpi_size * sizeof(int));
    int *displs = (int *)malloc(mpi_size * sizeof(int));
    for (int i = 0; i < mpi_size; ++i) {
        if (i < remaining_rows) {
            sendcounts[i] = (rows_per_process + 1) * K;
            displs[i] = (i * rows_per_process + i) * K;
        } else {
            sendcounts[i] = rows_per_process * K;
            displs[i] = (i * rows_per_process + remaining_rows) * K;
        }
    }

    MPI_Scatterv(A, sendcounts, displs, MPI_FLOAT, sub_A, (end_row - start_row) * K, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    matmul_multithread(sub_A, B, sub_C, end_row - start_row, N, K, num_threads);

    for (int i = 0; i < mpi_size; ++i) {
        if (i < remaining_rows) {
            sendcounts[i] = (rows_per_process + 1) * N;
            displs[i] = (i * rows_per_process + i) * N;
        } else {
            sendcounts[i] = rows_per_process * N;
            displs[i] = (i * rows_per_process + remaining_rows) * N;
        }
    }

    MPI_Gatherv(sub_C, (end_row - start_row) * N, MPI_FLOAT, C, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    free(sendcounts);
    free(displs);
}


void matmul( float *A,  float *B, float *C, int M, int N, int K, int num_threads, int mpi_rank, int mpi_size) {
    
    matmul_distributed(A, B, C, M, N, K, num_threads, mpi_rank, mpi_size);
   
}