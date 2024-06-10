#define size 32
__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
    int global_x = get_group_id(0) * size; 
    int global_y = get_group_id(1) * size;
    
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    __local float sub_A[size][size];
    __local float sub_B[size][size];

    int row = global_y + local_y;
    int col = global_x + local_x;

    float sum = 0;
    for (int t = 0; t < (K + size - 1) / size; t++) {
        int x = t * size + local_x;
        int y = t * size + local_y;
        if (row < M && x < K) {
            sub_A[local_y][local_x] = A[row * K + x];
        } else {
            sub_A[local_y][local_x] = 0;
        }

        if (y < K && col < N) {
            sub_B[local_y][local_x] = B[y * N + col];
        } else {
            sub_B[local_y][local_x] = 0;
        }


        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < size; k++) {
            sum += sub_A[local_y][k] * sub_B[k][local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
    
}

__kernel void sgemm_stable(__global float *A, __global float *B, __global float *C, int M, int N, int K) {

  int row = get_global_id(0);
  int col = get_global_id(1);

  if(row < M && col < N) {
    float sum = 0.0;
    for(int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}
