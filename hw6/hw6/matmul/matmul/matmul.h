#pragma once

void matmul_initialize(int M, int N, int K);
void matmul(float *A, float *B, float *C, int M, int N, int K);
void matmul_finalize();
