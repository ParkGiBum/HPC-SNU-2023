#include <mpi.h>
#include <stdio.h>

#include "monte_carlo.h"
#include "util.h"

double monte_carlo(double *xs, double *ys, int num_points, int mpi_rank, int mpi_world_size, int threads_per_process) {
  int local_count = 0, global_count = 0;


  int local_num_point = num_points / mpi_world_size;
  int start = mpi_rank * local_num_point;
  int end = start + local_num_point;
  if(end > num_points) end = num_points ;
  //if (mpi_rank == 0){
  MPI_Bcast(xs, num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(ys, num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //}
  // Parallelize using OpenMP within the MPI process
  #pragma omp parallel for num_threads(threads_per_process) reduction(+:local_count)
  for (int i = start; i < end; i++) {
    double x = xs[i];
    double y = ys[i];

    if (x * x + y * y <= 1) {
      local_count++;
    }

  }
  //printf("global_point: %d\n", global_point);
  // printf( "local_count: %d\n", local_count);
  // printf("num_points: %d\n", start-end);
  // // Gather the counts from all processes
  MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // Rank 0 computes and returns the final estimate of PI
  if (mpi_rank == 0) {
    return (double)4 * global_count / num_points;
  } else {
    // Other ranks return a placeholder value
    return 0.00;
  }
}
double leibniz(double *xs, double *ys, int num_points, int mpi_rank, int mpi_world_size, int threads_per_process) {
  
  double pi = 0;
  for (int i = 0; i < 1000000000; i++) {
    if(i %2 == 0)
      pi += (double)1 /(double)( 2 * i + 1);
    else
      pi -= (double)1 /(double)( 2 * i + 1);
  }
  

  return pi * 4;
}

