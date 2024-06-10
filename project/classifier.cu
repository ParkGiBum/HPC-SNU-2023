#include <math.h>
#include <mpi.h>
#include <cassert>

#include "classifier.h"
#include "util.h"

#define BLOCK 32
#define BLOCK_SIZE 256
static int mpi_rank, mpi_size;
#define TILE_WIDTH 1024
static int batch_size = 256;

#define NGPU 4
const int NUM_GPUS = 4;
cudaStream_t streams[NUM_GPUS];

// Global pointers for device memory
static float *d_input_global[NUM_GPUS];
static float *d_weight_global[NUM_GPUS];
static float *d_bias_global[NUM_GPUS];
static float *d_output_global[NUM_GPUS];


#define CHECK_CUDA(call) { gpuAssert((call), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


int calculate_parameter_size() {
    int size = 0;

    // Add sizes for convolutional layers' weights and biases
    size += 256 * 70 * 7 * sizeof(float); // Size of w_conv1
    size += 256 * sizeof(float);          // Size of b_conv1
    size += 256 * 1008 * sizeof(float);   // Size of gamma_conv1
    size += 256 * 1008 * sizeof(float);   // Size of beta_conv1
    size += 256 * 256 * 7 * sizeof(float); // Size of w_conv2
    size += 256 * sizeof(float);          // Size of b_conv2
    size += 256 * 256 * 3 * sizeof(float); // Size of w_conv3
    size += 256 * sizeof(float);          // Size of b_conv3
    size += 256 * 256 * 3 * sizeof(float); // Size of w_conv4
    size += 256 * sizeof(float);          // Size of b_conv4
    size += 256 * 256 * 3 * sizeof(float); // Size of w_conv5
    size += 256 * sizeof(float);          // Size of b_conv5
    size += 256 * 256 * 3 * sizeof(float); // Size of w_conv6
    size += 256 * sizeof(float);          // Size of b_conv6
    size += 256 * 102 * sizeof(float);    // Size of gamma_conv6
    size += 256 * 102 * sizeof(float);    // Size of beta_conv6


    size += 8704 * 1024 * sizeof(float);  // Size of w_fc1
    size += 1024 * sizeof(float);         // Size of b_fc1
    size += 1024 * 1024 * sizeof(float);  // Size of w_fc2
    size += 1024 * sizeof(float);         // Size of b_fc2
    size += 1024 * 4 * sizeof(float);     // Size of w_fc3
    size += 4 * sizeof(float);            // Size of b_fc3

    return size;
}

// Multi-dimensional matrix containing fp32 elements
struct Tensor {
  Tensor(std::vector<int> shape_);
  Tensor(std::vector<int> shape_, float *buf_);
  ~Tensor();
  int num_elem();
  void fill_zeros();

  float *buf = nullptr;
  int ndim = 0;
  int shape[4];
};

Tensor::Tensor(std::vector<int> shape_) {
  ndim = shape_.size();
  for (int i = 0; i < ndim; ++i) { shape[i] = shape_[i]; }
  int N_ = num_elem();
  buf = (float *) calloc(N_, sizeof(float));
}

Tensor::Tensor(std::vector<int> shape_, float *buf_) {
  ndim = shape_.size();
  for (int i = 0; i < ndim; ++i) { shape[i] = shape_[i]; }
  int N_ = num_elem();
  buf = (float *) calloc(N_, sizeof(float));
  for (int n = 0; n < N_; ++n) { buf[n] = buf_[n]; }
}

Tensor::~Tensor() {
  if (buf != nullptr) free(buf);
}

int Tensor::num_elem() {
  int sz = 1;
  for (int i = 0; i < ndim; ++i) { sz *= shape[i]; }
  return sz;
}

void Tensor::fill_zeros() {
  int N_ = num_elem();
  for (int n = 0; n < N_; ++n) { buf[n] = 0.0; }
}

// Parameters
Tensor *w_conv1, *w_conv2, *w_conv3, *w_conv4, *w_conv5, *w_conv6, *b_conv1,
    *b_conv2, *b_conv3, *b_conv4, *b_conv5, *b_conv6, *w_fc1, *w_fc2, *w_fc3,
    *b_fc1, *b_fc2, *b_fc3, *gamma_conv1, *beta_conv1, *gamma_conv6, *beta_conv6;

// Activations
Tensor *a_conv1, *a_layernorm1, *a_relu1, *a_pool1;
Tensor *a_conv2, *a_relu2, *a_pool2;
Tensor *a_conv3, *a_relu3;
Tensor *a_conv4, *a_relu4;
Tensor *a_conv5, *a_relu5;
Tensor *a_conv6, *a_layernorm6, *a_relu6, *a_pool6;
Tensor *a_collapse;
Tensor *a_linear1, *a_relu7;
Tensor *a_linear2, *a_relu8;
Tensor *a_linear3;

// Operations




__global__ void relu_kernel(float *input, float *output, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}


void linear_gpu(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output, bool has_bias);
__global__ void linear_kernel(float *input, float *weight, float *bias, float *output,  int input_channels, int output_channels, int batch, bool has_bias) ;
void conv1d_gpu(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output, int stride, int padding, int dilation, bool has_bias);
__global__ void conv1d_kernel(float *input, float *weight, float *bias, float *output, int in_channels, int out_channels, int kernel_size, int input_length, int output_length, int batch,  int stride, int padding, bool has_bias) ;
__global__ void relu_kernel(float *input, float *output, int features, int batch) ;
void relu_gpu(Tensor *input, Tensor *output);

void collapse(Tensor *input, Tensor *output);
void linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            bool has_bias);
void relu(Tensor *input, Tensor *output);
void maxpool1d(Tensor *input, Tensor *output, int kernel_size, int stride);
void layernorm(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *output);
__global__ void maxpool1d_kernel(float *input, float *output, int kernel_size, int stride, int IC, int IL, int OC, int OL, int batch);
void maxpool1d_gpu(Tensor *input, Tensor *output, int kernel_size, int stride) ;
__global__ void collapse_kernel(float *input, float *output, int features, int batch);
void collapse_gpu(Tensor *input, Tensor *output);


// Only the first process (root, mpi_rank == 
void classifier(float *input_, float *output_, int N) {

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    // printf("mpi size is %d",mpi_size);

    int num_nodes = mpi_size; // Number of nodes
    int num_gpus_per_node = mpi_size; // Number of GPUs per node


    int local_N = N / mpi_size;
    float *local_input = new float[local_N  * VOCAB_SIZE * MAX_LENGTH];
    float *local_output = new float[local_N ];

    
    
    if (mpi_rank == 0)
        MPI_Scatter(input_, local_N * VOCAB_SIZE * MAX_LENGTH, MPI_FLOAT,
                    local_input, local_N * VOCAB_SIZE * MAX_LENGTH, MPI_FLOAT,
                    0, MPI_COMM_WORLD);
    else
        MPI_Scatter(NULL, local_N * VOCAB_SIZE * MAX_LENGTH, MPI_FLOAT,
                    local_input, local_N * VOCAB_SIZE * MAX_LENGTH, MPI_FLOAT,
                    0, MPI_COMM_WORLD);
                    
  
//   if (mpi_rank == 0) {
    for (int n = 0; n < local_N/batch_size; ++n) {  // N input sentences

        // Load one input sentence from input
        Tensor *one_input = new Tensor({batch_size, VOCAB_SIZE, MAX_LENGTH}, local_input + batch_size * n * VOCAB_SIZE * MAX_LENGTH);
        
        // Conv block 1 : Conv1d + LayerNorm + ReLU + MaxPool1d
        conv1d_gpu(one_input, w_conv1, b_conv1, a_conv1, 1, 0, 1, true);
        layernorm(a_conv1, gamma_conv1, beta_conv1, a_layernorm1);
        relu(a_layernorm1, a_relu1);
        maxpool1d(a_relu1, a_pool1, 3, 3);

        // Conv block 2 : Conv1d + ReLU + MaxPool1d
        conv1d_gpu(a_pool1, w_conv2, b_conv2, a_conv2, 1, 0, 1, true);
        relu(a_conv2, a_relu2);
        maxpool1d(a_relu2, a_pool2, 3, 3);

        // Conv block 3 : Conv1d + ReLU
        conv1d_gpu(a_pool2, w_conv3, b_conv3, a_conv3, 1, 0, 1, true);
        relu(a_conv3, a_relu3);

        // Conv block 4 : Conv1d + ReLU
        conv1d_gpu(a_relu3, w_conv4, b_conv4, a_conv4, 1, 0, 1, true);
        relu(a_conv4, a_relu4);

        // Conv block 5 : Conv1d + ReLU
        conv1d_gpu(a_relu4, w_conv5, b_conv5, a_conv5, 1, 0, 1, true);
        relu(a_conv5, a_relu5);

        // Conv block 6 : Conv1d + LayerNorm + ReLU + MaxPool1d
        conv1d_gpu(a_relu5, w_conv6, b_conv6, a_conv6, 1, 0, 1, true);
        layernorm(a_conv6, gamma_conv6, beta_conv6, a_layernorm6);
        relu(a_layernorm6, a_relu6);
        maxpool1d(a_relu6, a_pool6, 3, 3);

        // Collapse
        collapse_gpu(a_pool6, a_collapse);

        // FC block 1 : Linear + ReLU
        linear_gpu(a_collapse, w_fc1, b_fc1, a_linear1, true);
        relu(a_linear1, a_relu7);

        // FC block 2 : Linear + ReLU
        linear_gpu(a_relu7, w_fc2, b_fc2, a_linear2, true);
        relu(a_linear2, a_relu8);

        // FC block 3 : Linear
        linear_gpu(a_relu8, w_fc3, b_fc3, a_linear3, true);

        for (int k = 0; k < batch_size; ++k) {
            float max_val = -1e99f;
            int max_idx = 0;
            for (int i = 0; i < a_linear3->num_elem() / batch_size; ++i) {
                float current_val = a_linear3->buf[k * (a_linear3->num_elem() / batch_size) + i];
                if (current_val > max_val) {
                    max_val = current_val;
                    max_idx = i;
                }
            }

            local_output[n * batch_size + k] = max_idx;
        }
        
    }   
    if (mpi_rank == 0)
        MPI_Gather(local_output, local_N , MPI_FLOAT,output_, local_N , MPI_FLOAT,0, MPI_COMM_WORLD);
    else
        MPI_Gather(local_output, local_N , MPI_FLOAT, NULL, local_N , MPI_FLOAT,0, MPI_COMM_WORLD);

    delete[] local_input;
    delete[] local_output;
}

void allocate_and_copy_to_gpu(Tensor **tensor, const std::vector<int>& shape, float *host_data, cudaStream_t stream) {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }

    // Allocate memory on GPU
    float *device_data;
    cudaMalloc(&device_data, size * sizeof(float));

    // Asynchronously copy data from host to device
    cudaMemcpyAsync(device_data, host_data, size * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Create a new Tensor object with device data
    *tensor = new Tensor(shape, device_data);
}


void initialize_classifier(float *parameter, int N) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    
    int param_size = 0;

    // printf("number of gpu is %d \n\n" , num_gpus);
    // int size_of_parameter = calculate_parameter_size();
    // printf("size of parameter is %d \n\n" , size_of_parameter);
    // Calculate the maximum sizes needed for weights, biases, inputs, and outputs
    int max_weight_size = sizeof(float) * 1024 * 8704; // Size of w_fc1
    int max_bias_size = sizeof(float) * 1024;         // Size of b_fc3
    int max_input_size = sizeof(float) * batch_size * 256 * 1008 / num_gpus; // Size of a_collapse
    int max_output_size = sizeof(float) * batch_size * 256 * 1008 / num_gpus; // Calculate based on your largest output tensor

    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        // Allocate memory on GPU
        CHECK_CUDA(cudaMalloc(&d_input_global[i], max_input_size));
        CHECK_CUDA(cudaMalloc(&d_weight_global[i], max_weight_size));
        CHECK_CUDA(cudaMalloc(&d_bias_global[i], max_bias_size));
        CHECK_CUDA(cudaMalloc(&d_output_global[i], max_output_size));
    }
    if (mpi_rank == 0) {
        
        w_conv1 = new Tensor({256, 70, 7}, parameter + OFFSET0);
        param_size += 256 * 70 * 7 ; // Size of w_conv1
        b_conv1 = new Tensor({256}, parameter + OFFSET1);
        param_size += 256 ;          // Size of b_conv1
        gamma_conv1 = new Tensor({256, 1008}, parameter + OFFSET2);
        param_size += 256 * 1008 ;   // Size of gamma_conv1
        beta_conv1 = new Tensor({256, 1008}, parameter + OFFSET3);
        param_size += 256 * 1008 ;   // Size of beta_conv1
        w_conv2 = new Tensor({256, 256, 7}, parameter + OFFSET4);
        param_size += 256 * 256 * 7 ; // Size of w_conv2
        b_conv2 = new Tensor({256}, parameter + OFFSET5);
        param_size += 256 ;          // Size of b_conv2
        w_conv3 = new Tensor({256, 256, 3}, parameter + OFFSET6);
        param_size += 256 * 256 * 3 ; // Size of w_conv3
        b_conv3 = new Tensor({256}, parameter + OFFSET7);
        param_size += 256 ;          // Size of b_conv3
        w_conv4 = new Tensor({256, 256, 3}, parameter + OFFSET8);
        param_size += 256 * 256 * 3 ; // Size of w_conv4
        b_conv4 = new Tensor({256}, parameter + OFFSET9);
        param_size += 256 ;          // Size of b_conv4
        w_conv5 = new Tensor({256, 256, 3}, parameter + OFFSET10);
        param_size += 256 * 256 * 3 ; // Size of w_conv5
        b_conv5 = new Tensor({256}, parameter + OFFSET11);
        param_size += 256 ;          // Size of b_conv5
        w_conv6 = new Tensor({256, 256, 3}, parameter + OFFSET12);
        param_size += 256 * 256 * 3 ; // Size of w_conv6
        b_conv6 = new Tensor({256}, parameter + OFFSET13);
        param_size += 256 ;          // Size of b_conv6
        gamma_conv6 = new Tensor({256, 102}, parameter + OFFSET14);
        param_size += 256 * 102 ;    // Size of gamma_conv6
        beta_conv6 = new Tensor({256, 102}, parameter + OFFSET15);
        param_size += 256 * 102 ;    // Size of beta_conv6
        w_fc1 = new Tensor({1024, 8704}, parameter + OFFSET16);
        param_size += 8704 * 1024 ;  // Size of w_fc1
        b_fc1 = new Tensor({1024}, parameter + OFFSET17);
        param_size += 1024 ;         // Size of b_fc1
        w_fc2 = new Tensor({1024, 1024}, parameter + OFFSET18);
        param_size += 1024 * 1024 ;  // Size of w_fc2
        b_fc2 = new Tensor({1024}, parameter + OFFSET19);
        param_size += 1024 ;         // Size of b_fc2
        w_fc3 = new Tensor({4, 1024}, parameter + OFFSET20);
        param_size += 1024 * 4 ;     // Size of w_fc3
        b_fc3 = new Tensor({4}, parameter + OFFSET21);
        param_size += 4 ;   
         // Size of b_fc3

        a_conv1 = new Tensor({batch_size,  256, 1008});
        a_layernorm1 = new Tensor({batch_size,  256, 1008});
        a_relu1 = new Tensor({batch_size,  256, 1008});
        a_pool1 = new Tensor({batch_size,  256, 336});
        a_conv2 = new Tensor({batch_size,  256, 330});
        a_relu2 = new Tensor({batch_size,  256, 330});
        a_pool2 = new Tensor({batch_size,  256, 110});
        a_conv3 = new Tensor({batch_size,  256, 108});
        a_relu3 = new Tensor({batch_size,  256, 108});
        a_conv4 = new Tensor({batch_size,  256, 106});
        a_relu4 = new Tensor({batch_size,  256, 106});
        a_conv5 = new Tensor({batch_size,  256, 104});
        a_relu5 = new Tensor({batch_size,  256, 104});
        a_conv6 = new Tensor({batch_size,  256, 102});
        a_layernorm6 = new Tensor({batch_size,  256, 102});
        a_relu6 = new Tensor({batch_size,  256, 102});
        a_pool6 = new Tensor({batch_size,  256, 34});
        a_collapse = new Tensor({batch_size,  8704});
        a_linear1 = new Tensor({batch_size,  1024});
        a_relu7 = new Tensor({batch_size,  1024});
        a_linear2 = new Tensor({batch_size,  1024});
        a_relu8 = new Tensor({batch_size,  1024});
        a_linear3 = new Tensor({batch_size,  4});
  }

    // MPI_Bcast(parameter, param_size * sizeof(float), MPI_CHAR, 0, MPI_COMM_WORLD);


    if(mpi_rank != 0){
        w_conv1 = new Tensor({256, 70, 7});
        b_conv1 = new Tensor({256});
        gamma_conv1 = new Tensor({256, 1008});
        beta_conv1 = new Tensor({256, 1008});
        w_conv2 = new Tensor({256, 256, 7});
        b_conv2 = new Tensor({256});
        w_conv3 = new Tensor({256, 256, 3});
        b_conv3 = new Tensor({256});
        w_conv4 = new Tensor({256, 256, 3});
        b_conv4 = new Tensor({256});
        w_conv5 = new Tensor({256, 256, 3});
        b_conv5 = new Tensor({256});
        w_conv6 = new Tensor({256, 256, 3});
        b_conv6 = new Tensor({256});
        gamma_conv6 = new Tensor({256, 102});
        beta_conv6 = new Tensor({256, 102});
        w_fc1 = new Tensor({1024, 8704});
        b_fc1 = new Tensor({1024});
        w_fc2 = new Tensor({1024, 1024});
        b_fc2 = new Tensor({1024});
        w_fc3 = new Tensor({4, 1024});
        b_fc3 = new Tensor({4});
        

        a_conv1 = new Tensor({batch_size,  256, 1008});
        a_layernorm1 = new Tensor({batch_size,  256, 1008});
        a_relu1 = new Tensor({batch_size,  256, 1008});
        a_pool1 = new Tensor({batch_size,  256, 336});
        a_conv2 = new Tensor({batch_size,  256, 330});
        a_relu2 = new Tensor({batch_size,  256, 330});
        a_pool2 = new Tensor({batch_size,  256, 110});
        a_conv3 = new Tensor({batch_size,  256, 108});
        a_relu3 = new Tensor({batch_size,  256, 108});
        a_conv4 = new Tensor({batch_size,  256, 106});
        a_relu4 = new Tensor({batch_size,  256, 106});
        a_conv5 = new Tensor({batch_size,  256, 104});
        a_relu5 = new Tensor({batch_size,  256, 104});
        a_conv6 = new Tensor({batch_size,  256, 102});
        a_layernorm6 = new Tensor({batch_size,  256, 102});
        a_relu6 = new Tensor({batch_size,  256, 102});
        a_pool6 = new Tensor({batch_size,  256, 34});
        a_collapse = new Tensor({batch_size,  8704});
        a_linear1 = new Tensor({batch_size,  1024});
        a_relu7 = new Tensor({batch_size,  1024});
        a_linear2 = new Tensor({batch_size,  1024});
        a_relu8 = new Tensor({batch_size,  1024});
        a_linear3 = new Tensor({batch_size,  4});

    }
    //broadcasting

    MPI_Bcast(w_conv1->buf, 256 * 70 * 7, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_conv1->buf, 256, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(gamma_conv1->buf, 256 * 1008, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(beta_conv1->buf, 256 * 1008, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(w_conv2->buf, 256 * 256 * 7, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_conv2->buf, 256, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(w_conv3->buf, 256 * 256 * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_conv3->buf, 256, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(w_conv4->buf, 256 * 256 * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_conv4->buf, 256, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(w_conv5->buf, 256 * 256 * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_conv5->buf, 256, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(w_conv6->buf, 256 * 256 * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_conv6->buf, 256, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(gamma_conv6->buf, 256 * 102, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(beta_conv6->buf, 256 * 102, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(w_fc1->buf, 8704 * 1024, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_fc1->buf, 1024, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(w_fc2->buf, 1024 * 1024, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_fc2->buf, 1024, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(w_fc3->buf, 1024 * 4, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_fc3->buf, 4, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Allocate memory on GPU and copy data from host to device
    

}

// Free all dynamically allocated variables
void finalize_classifier() {
  if (1) {
    delete w_conv1;
    delete b_conv1;
    delete w_conv2;
    delete b_conv2;
    delete w_conv3;
    delete b_conv3;
    delete w_conv4;
    delete b_conv4;
    delete w_conv5;
    delete b_conv5;
    delete w_conv6;
    delete b_conv6;
    delete w_fc1;
    delete b_fc1;
    delete w_fc2;
    delete b_fc2;
    delete w_fc3;
    delete b_fc3;
    delete gamma_conv1;
    delete gamma_conv6;
    delete beta_conv1;
    delete beta_conv6;
    delete a_conv1;
    delete a_layernorm1;
    delete a_relu1;
    delete a_pool1;
    delete a_conv2;
    delete a_relu2;
    delete a_pool2;
    delete a_conv3;
    delete a_relu3;
    delete a_conv4;
    delete a_relu4;
    delete a_conv5;
    delete a_relu5;
    delete a_conv6;
    delete a_layernorm6;
    delete a_relu6;
    delete a_pool6;
    delete a_collapse;
    delete a_linear1;
    delete a_relu7;
    delete a_linear2;
    delete a_relu8;
    delete a_linear3;
  }
}



void relu(Tensor *input, Tensor *output) {
    int batch = input->shape[0];
    int features = input->num_elem() / batch; // Total elements divided by batch size

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < features; ++i) {
            int idx = b * features + i;
            if (input->buf[idx] > 0.0f)
                output->buf[idx] = input->buf[idx];
            else
                output->buf[idx] = 0.0f;
        }
    }
}


void maxpool1d(Tensor *input, Tensor *output, int kernel_size, int stride) {
    int batch = input->shape[0]; // New: batch dimension
    int IC = input->shape[1];    // Input Channels
    int IL = input->shape[2];    // Input Length
    int OC = output->shape[1];   // Output Channels
    int OL = output->shape[2];   // Output Length

    for (int b = 0; b < batch; ++b) { // Loop over batches
        for (int oc = 0; oc < OC; ++oc) {
            for (int ol = 0; ol < OL; ++ol) {
                float mx = -1e99;
                for (int ks = 0; ks < kernel_size; ++ks) {
                    int input_index = b * IC * IL + oc * IL + ks + ol * stride;
                    if (input_index < b * IC * IL + oc * IL + IL) { // Check within bounds
                        float val = input->buf[input_index];
                        if (val > mx) mx = val;
                    }
                }
                output->buf[b * OC * OL + oc * OL + ol] = mx;
            }
        }
    }
}

void collapse(Tensor *input, Tensor *output) {
    int batch = input->shape[0];
    int features = input->num_elem() / batch;

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < features; ++i) {
            int idx = b * features + i;
            output->buf[idx] = input->buf[idx];
        }
    }
}



void layernorm_1(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *output) {
  // E[X], E[X^2]
  float sum1 = 0.0f, sum2 = 0.0f;
  for (int i = 0; i < input->num_elem(); ++i) {
      sum1 += input->buf[i];
      sum2 += input->buf[i] * input->buf[i];
  }
  float mean1 = sum1 / (float)input->num_elem();
  float mean2 = sum2 / (float)input->num_elem();

  // V[X]
  float var = mean2 - mean1 * mean1; 

  // Normalization
  for (int i = 0; i < input->num_elem(); ++i) {
    output->buf[i] = (input->buf[i] - mean1) / sqrtf(var + 1e-5) * gamma->buf[i] + beta->buf[i];
  }
}

void layernorm(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *output) {
    int batch = input->shape[0];
    int feature_size = input->shape[1] * input->shape[2]; // Assuming normalization over last two dimensions

    for (int b = 0; b < batch; ++b) {
        float sum1 = 0.0f, sum2 = 0.0f;
        int start = b * feature_size;
        int end = start + feature_size;

        // Calculate mean and variance for this batch
        for (int i = start; i < end; ++i) {
            sum1 += input->buf[i];
            sum2 += input->buf[i] * input->buf[i];
        }
        float mean = sum1 / feature_size;
        float var = (sum2 / feature_size) - (mean * mean);

        // Apply normalization for this batch
        for (int i = start; i < end; ++i) {
            int gamma_beta_index = i % feature_size; // gamma and beta are repeated for each batch
            output->buf[i] = (input->buf[i] - mean) / sqrtf(var + 1e-5) * gamma->buf[gamma_beta_index] + beta->buf[gamma_beta_index];
        }
    }
}


// Function to transform input data using im2col
// void im2col(const float* input, float* data_col,
//             int channels, int input_length, int kernel_size,
//             int stride, int padding, int output_length) {
//     int col_length = kernel_size * channels;
//     for (int c = 0; c < col_length; ++c) {
//         int w_offset = c % kernel_size;
//         int c_im = c / kernel_size;
//         for (int h = 0; h < output_length; ++h) {
//             int h_pad = h * stride - padding + w_offset;
//             if (h_pad >= 0 && h_pad < input_length) {
//                 data_col[(c * output_length) + h] = input[(c_im * input_length) + h_pad];
//             } else {
//                 data_col[(c * output_length) + h] = 0;
//             }
//         }
//     }
// }

__global__ void conv1d_kernel(float *input, float *weight, float *bias, float *output,
                                    int in_channels, int out_channels, int kernel_size,
                                    int input_length, int output_length, int batch,
                                    int stride, int padding, bool has_bias) {
    extern __shared__ float sharedMem[];

    int tileRow = blockIdx.x * blockDim.x + threadIdx.x;  // Tile row index for output channels
    int tileCol = blockIdx.y * blockDim.y + threadIdx.y;  // Tile column index for output length
    int batchIdx = blockIdx.z * blockDim.z + threadIdx.z; // Batch index

    int oc = tileRow;   // Output channel index
    int ol = tileCol;   // Output location index

    // Shared memory for input and weight tiles
    float* inputTile = &sharedMem[0];
    float* weightTile = &sharedMem[kernel_size * in_channels];

    if (oc < out_channels && ol < output_length && batchIdx < batch) {
        float val = 0.0f;

        // Load a tile of the input and weight into shared memory
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int ks = 0; ks < kernel_size; ++ks) {
                int input_index = ol * stride - padding + ks;
                if (input_index >= 0 && input_index < input_length) {
                    inputTile[ic * kernel_size + ks] = input[batchIdx * in_channels * input_length + ic * input_length + input_index];
                    weightTile[oc * in_channels * kernel_size + ic * kernel_size + ks] = weight[oc * in_channels * kernel_size + ic * kernel_size + ks];
                }
            }
        }

        __syncthreads(); // Synchronize to make sure tiles are loaded

        // Compute using the tiles
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int ks = 0; ks < kernel_size; ++ks) {
                val += weightTile[oc * in_channels * kernel_size + ic * kernel_size + ks] * inputTile[ic * kernel_size + ks];
            }
        }
        // printf("val is %f \n", val);

        if (has_bias) val += bias[oc];
        output[batchIdx * out_channels * output_length + oc * output_length + ol] = val;
    }
}

void conv1d_gpu(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
                int stride, int padding, int dilation, bool has_bias) {
    // Calculate output dimensions
    int num_gpus = NUM_GPUS;
    //cudaGetDeviceCount(&num_gpus);
    

    // Calculate output dimensions
    int out_channels = weight->shape[0];
    int in_channels = weight->shape[1];
    int kernel_size = weight->shape[2];
    int input_length = input->shape[2];
    int batch = input->shape[0];
    int output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Step 2: Divide batches among GPUs
    int batches_per_gpu = batch / num_gpus;

    // Loop over GPUs
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        int start_batch = gpu * batches_per_gpu;
        int end_batch = min(start_batch + batches_per_gpu, batch);

        // Asynchronous copy data from CPU to GPU using streams
        cudaMemcpyAsync(d_input_global[gpu], input->buf + start_batch * in_channels * input_length, 
                        (end_batch - start_batch) * sizeof(float) * in_channels * input_length, cudaMemcpyHostToDevice, streams[gpu]);
        cudaMemcpyAsync(d_weight_global[gpu], weight->buf, sizeof(float) * out_channels * in_channels * kernel_size, cudaMemcpyHostToDevice, streams[gpu]);
        cudaMemcpyAsync(d_bias_global[gpu], bias->buf, sizeof(float) * out_channels, cudaMemcpyHostToDevice, streams[gpu]);

    }
    // Loop over GPUs
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        int start_batch = gpu * batches_per_gpu;
        int end_batch = min(start_batch + batches_per_gpu, batch);

        cudaSetDevice(gpu);

        
        dim3 threadsPerBlock(BLOCK, BLOCK, 1); 
        dim3 numBlocks((out_channels + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_length + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (end_batch - start_batch + threadsPerBlock.z - 1) / threadsPerBlock.z);

        // Launch the kernel with the stream
        conv1d_kernel<<<numBlocks, threadsPerBlock, 0, streams[gpu]>>>(d_input_global[gpu], d_weight_global[gpu], d_bias_global[gpu], d_output_global[gpu],
                                                                       in_channels, out_channels, kernel_size,
                                                                       input_length, output_length, end_batch - start_batch,
                                                                       stride, padding, has_bias);

    }

    // Wait for all async jobs to finish
    for (int i = 0; i < num_gpus; ++i) {
        int start_batch = i * batches_per_gpu;
        int end_batch = min(start_batch + batches_per_gpu, batch);
        cudaMemcpyAsync(output->buf + start_batch * out_channels * output_length, d_output_global[i ], 
                        (end_batch - start_batch) * sizeof(float) * out_channels * output_length, cudaMemcpyDeviceToHost, streams[i ]);
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
}

__global__ void linear_kernel(float *input, float *weight, float *bias, float *output,
                              int input_channels, int output_channels, int batch, bool has_bias) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;

    int row = bx * blockDim.x + tx;
    int batchIdx = by;

    if (row < output_channels && batchIdx < batch) {
        float val = 0.0f;
        for (int ic = 0; ic < input_channels; ++ic) {
            val += input[batchIdx * input_channels + ic] * weight[row * input_channels + ic];
        }
        if (has_bias) {
            val += bias[row];
        }
        output[batchIdx * output_channels + row] = val;
    }
}

void linear_gpu(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output, bool has_bias) {
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 1) {
        // Handle error: no GPU found
        return;
    }

    int batch = input->shape[0];
    int input_channels = input->shape[1];
    int output_channels = output->shape[1];
    int batches_per_gpu = (batch + num_gpus - 1) / num_gpus;  // Round up to distribute work more evenly

    // Assuming global device memory pointers are available
    extern float *d_input_global[NUM_GPUS];
    extern float *d_weight_global[NUM_GPUS];
    extern float *d_bias_global[NUM_GPUS];
    extern float *d_output_global[NUM_GPUS];
    extern cudaStream_t streams[NUM_GPUS];

    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(gpu);

        int start_batch = gpu * batches_per_gpu;
        int end_batch = min(start_batch + batches_per_gpu, batch);

        // Asynchronous copy of data to device memory
        cudaMemcpyAsync(d_input_global[gpu], input->buf + start_batch * input_channels, 
                        (end_batch - start_batch) * sizeof(float) * input_channels, cudaMemcpyHostToDevice, streams[gpu]);
        cudaMemcpyAsync(d_weight_global[gpu], weight->buf, sizeof(float) * input_channels * output_channels, cudaMemcpyHostToDevice, streams[gpu]);
        if (has_bias) {
            cudaMemcpyAsync(d_bias_global[gpu], bias->buf, sizeof(float) * output_channels, cudaMemcpyHostToDevice, streams[gpu]);
        }
    }

    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(gpu);

        int start_batch = gpu * batches_per_gpu;
        int end_batch = min(start_batch + batches_per_gpu, batch);

        dim3 threadsPerBlock(TILE_WIDTH, 1);
        dim3 numBlocks((output_channels + TILE_WIDTH - 1) / TILE_WIDTH, end_batch - start_batch);
        linear_kernel<<<numBlocks, threadsPerBlock, 0, streams[gpu]>>>(d_input_global[gpu], d_weight_global[gpu], d_bias_global[gpu], d_output_global[gpu],
                                                                       input_channels, output_channels, end_batch - start_batch, has_bias);
    }

    // Wait for all async jobs to finish and copy results back to host
    for (int i = 0; i < num_gpus; ++i) {
        int start_batch = i * batches_per_gpu;
        int end_batch = min(start_batch + batches_per_gpu, batch);
        cudaMemcpyAsync(output->buf + start_batch * output_channels, d_output_global[i], 
                        (end_batch - start_batch) * sizeof(float) * output_channels, cudaMemcpyDeviceToHost, streams[i]);
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
}

__global__ void relu_kernel(float *input, float *output, int features, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global index
    int batchIdx = blockIdx.y * blockDim.y + threadIdx.y; // Batch index

    if (idx < features && batchIdx < batch) {
        int global_idx = batchIdx * features + idx;
        output[global_idx] = input[global_idx] > 0.0f ? input[global_idx] : 0.0f;
    }
}
void relu_gpu(Tensor *input, Tensor *output) {
    int num_gpus = NUM_GPUS;
    // cudaGetDeviceCount(&num_gpus);

    int batch = input->shape[0];
    int features = input->num_elem() / batch; // Total elements divided by batch size
    int batches_per_gpu = batch / num_gpus;

    // Loop over GPUs
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        int start_batch = gpu * batches_per_gpu;
        int end_batch = min(start_batch + batches_per_gpu, batch);

        cudaMemcpyAsync(d_input_global[gpu], input->buf + start_batch * features, 
                        (end_batch - start_batch) * sizeof(float) * features, cudaMemcpyHostToDevice, streams[gpu]);

        cudaSetDevice(gpu);

        dim3 threadsPerBlock(256, 1); // Assuming 256 threads per block
        dim3 numBlocks((features + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (end_batch - start_batch + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Launch the kernel with the stream
        relu_kernel<<<numBlocks, threadsPerBlock, 0, streams[gpu]>>>(d_input_global[gpu], d_output_global[gpu], features, end_batch - start_batch);

    }

    // Wait for all async jobs to finish
    for (int i = 0; i < num_gpus; ++i) {
        int start_batch = i * batches_per_gpu;
        int end_batch = min(start_batch + batches_per_gpu, batch);
        cudaMemcpyAsync(output->buf + start_batch * features, d_output_global[i], 
                        (end_batch - start_batch) * sizeof(float) * features, cudaMemcpyDeviceToHost, streams[i]);
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
}



__global__ void maxpool1d_kernel(float *input, float *output, int kernel_size, int stride, int IC, int IL, int OC, int OL, int batch) {
    int ol = blockIdx.x * blockDim.x + threadIdx.x; // Output length index
    int oc = blockIdx.y * blockDim.y + threadIdx.y; // Output channel index
    int b = blockIdx.z * blockDim.z + threadIdx.z;  // Batch index

    if (b < batch && oc < OC && ol < OL) {
        float mx = -1e99;
        for (int ks = 0; ks < kernel_size; ++ks) {
            int input_index = b * IC * IL + oc * IL + ks + ol * stride;
            if (input_index < b * IC * IL + oc * IL + IL) { // Check within bounds
                float val = input[input_index];
                if (val > mx) mx = val;
            }
        }
        output[b * OC * OL + oc * OL + ol] = mx;
    }
}
void maxpool1d_gpu(Tensor *input, Tensor *output, int kernel_size, int stride) {
    int num_gpus = NUM_GPUS;
    // cudaGetDeviceCount(&num_gpus);

    int batch = input->shape[0];
    int IC = input->shape[1];
    int IL = input->shape[2];
    int OC = output->shape[1];
    int OL = output->shape[2];
    int batches_per_gpu = batch / num_gpus;

    // Loop over GPUs
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        int start_batch = gpu * batches_per_gpu;
        int end_batch = min(start_batch + batches_per_gpu, batch);

        cudaMemcpyAsync(d_input_global[gpu], input->buf + start_batch * IC * IL,
                        (end_batch - start_batch) * sizeof(float) * IC * IL, cudaMemcpyHostToDevice, streams[gpu]);

        cudaSetDevice(gpu);

        dim3 threadsPerBlock(256, 1, 1); // Adjust based on GPU capabilities
        dim3 numBlocks((OL + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (OC + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (end_batch - start_batch + threadsPerBlock.z - 1) / threadsPerBlock.z);

        // Launch the kernel with the stream
        maxpool1d_kernel<<<numBlocks, threadsPerBlock, 0, streams[gpu]>>>(d_input_global[gpu], d_output_global[gpu], kernel_size, stride, IC, IL, OC, OL, end_batch - start_batch);
    }

    // Wait for all async jobs to finish
    for (int i = 0; i < num_gpus; ++i) {
        int start_batch = i * batches_per_gpu;
        int end_batch = min(start_batch + batches_per_gpu, batch);
        cudaMemcpyAsync(output->buf + start_batch * OC * OL, d_output_global[i],
                        (end_batch - start_batch) * sizeof(float) * OC * OL, cudaMemcpyDeviceToHost, streams[i]);
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
}
__global__ void collapse_kernel(float *input, float *output, int features, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global index
    int batchIdx = blockIdx.y * blockDim.y + threadIdx.y; // Batch index

    if (idx < features && batchIdx < batch) {
        int global_idx = batchIdx * features + idx;
        output[global_idx] = input[global_idx];
    }
}
void collapse_gpu(Tensor *input, Tensor *output) {
    int num_gpus = NUM_GPUS;
    // cudaGetDeviceCount(&num_gpus);

    int batch = input->shape[0];
    int features = input->num_elem() / batch;
    int batches_per_gpu = batch / num_gpus;

    // Loop over GPUs
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        int start_batch = gpu * batches_per_gpu;
        int end_batch = min(start_batch + batches_per_gpu, batch);

        cudaMemcpyAsync(d_input_global[gpu], input->buf + start_batch * features,
                        (end_batch - start_batch) * sizeof(float) * features, cudaMemcpyHostToDevice, streams[gpu]);

        cudaSetDevice(gpu);

        dim3 threadsPerBlock(256, 1); // Adjust based on GPU capabilities
        dim3 numBlocks((features + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (end_batch - start_batch + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Launch the kernel with the stream
        collapse_kernel<<<numBlocks, threadsPerBlock, 0, streams[gpu]>>>(d_input_global[gpu], d_output_global[gpu], features, end_batch - start_batch);
    }

    // Wait for all async jobs to finish
    for (int i = 0; i < num_gpus; ++i) {
        int start_batch = i * batches_per_gpu;
        int end_batch = min(start_batch + batches_per_gpu, batch);
        cudaMemcpyAsync(output->buf + start_batch * features, d_output_global[i],
                        (end_batch - start_batch) * sizeof(float) * features, cudaMemcpyDeviceToHost, streams[i]);
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
}
