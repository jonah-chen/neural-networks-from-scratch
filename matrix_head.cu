#include "matrix.cuh"
/* This file will contain the constructor and destructor for the matrices
 */

// #define DEBUG

#include <algorithm>
#include <cstring>
#include <ctime>
#include <chrono>
#include <random>
#include <curand_kernel.h> // compile with -lcurand flag!
#include <cublas_v2.h>

#define BLOCK_SIZE 1024
#define DIVIDE(A,B) ((A+B-1)/B)
#define BLOCKS(N) DIVIDE(N,BLOCK_SIZE)

__global__ void set(float* matrix, const float value, const int size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
        matrix[i] = value;
}

__global__ void set_rand(float* matrix, const long long time, const int size, const float mean=0.0f, const float variance=1.0f)
{
    int i = 2 * (threadIdx.x + blockDim.x * blockIdx.x);

    if (i < size)
    {
        // initlaiize RNG
        curandState local;
        curand_init(time+i, 0, 0, &local);

        // generate 2 random numbers
        if (mean==0.0f&&variance==1.0f)
        {
            matrix[i] = curand_normal(&local);
            if (i+1 < size)
                matrix[i+1] = curand_normal(&local);
        }
        else
        {
            matrix[i] = curand_normal(&local) * variance + mean;
            if (i+1 < size)
                matrix[i+1] = curand_normal(&local) * variance + mean;
        }
    }
}

Matrix::Matrix(const int n, bool gpu) : Matrix::Matrix(n, 1, gpu) { }

Matrix::Matrix(const int n, const float val, bool gpu): Matrix::Matrix(n, 1, val, gpu) { }

Matrix::Matrix(const int m, const int n, bool gpu)
{
    dim1 = m;
    dim2 = n;
    gpu_enabled = gpu;

    if (gpu_enabled)
    {
        if (cudaMalloc((void**)&matrix, sizeof(float)*dim1*dim2) != cudaSuccess)
            throw "memory allocation failed\n";
        if (cudaMalloc((void**)&dummy, DUMMY_SIZE) != cudaSuccess)
            throw "memory allocation failed\n";

        std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);

        set_rand <<<BLOCKS(DIVIDE(dim1*dim2,2)), BLOCK_SIZE>>> (matrix, nanoseconds.count(), dim1*dim2);
    }
    else
    {
        matrix = new float[dim1*dim2];

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<float> d{0,1};
        for (int i = 0; i < dim1*dim2; ++i)
            matrix[i] = d(rd);
    }
}

Matrix::Matrix(const int m, const int n, const float val, bool gpu)
{
    dim1 = m;
    dim2 = n;
    gpu_enabled = gpu;

    if (gpu_enabled)
    {
        if (cudaMalloc((void**)&matrix, sizeof(float)*dim1*dim2) != cudaSuccess)
            throw "memory allocation failed\n";
        if (cudaMalloc((void**)&dummy, DUMMY_SIZE) != cudaSuccess)
            throw "memory allocation failed\n";

        set <<<BLOCKS(dim1*dim2), BLOCK_SIZE>>> (matrix, val, dim1*dim2);
    }
    else
    {
        matrix = new float[dim1*dim2]();
        if (val)
            std::fill_n(matrix, dim1*dim2, val);
    }
}

Matrix::Matrix(float* h_arr, const int size, bool gpu) : Matrix::Matrix(h_arr, size, 1, gpu) { }

Matrix::Matrix(float* h_arr, const int m, const int n, bool gpu)
{
    dim1 = m;
    dim2 = n;
    gpu_enabled = gpu;

    if (gpu_enabled)
    {
        if (cudaMalloc((void**)&matrix, sizeof(float)*dim1*dim2) != cudaSuccess)
            throw "memory allocation failed\n";
        if (cudaMalloc((void**)&dummy, DUMMY_SIZE) != cudaSuccess)
            throw "memory allocation failed\n";
        cudaMemcpy(matrix, h_arr, sizeof(float)*dim1*dim2, cudaMemcpyHostToDevice);
    }
    else
    {
        matrix = new float[dim1*dim2];
        std::memcpy(matrix, h_arr, sizeof(float)*dim1*dim2);
    }
}

Matrix::Matrix(float** h_arr, const int m, const int n, bool gpu)
{
    dim1 = m;
    dim2 = n;
    gpu_enabled = gpu;

    std::cout << "WARNING: Make sure you have an rectangular array. If you are using pointer arrays, this function may fail.\n";

    if (gpu_enabled)
    {
        if (cudaMalloc((void**)&matrix, sizeof(float)*dim1*dim2) != cudaSuccess)
            throw "memory allocation failed\n";
        cudaMemcpy(matrix, (void*)h_arr, sizeof(float)*dim1*dim2, cudaMemcpyHostToDevice);
    }
    else
    {
        matrix = new float[dim1*dim2];
        std::memcpy(matrix, (void*)h_arr, sizeof(float)*dim1*dim2);
    }
}

Matrix::~Matrix()
{
    if (gpu_enabled)
    {
        cudaFree(matrix);
        cudaFree(dummy);
    }
    else
        delete[] matrix;
}
/////////////////////////////////////////////
// testing

#ifdef DEBUG

int main(int argc, char* argv[])
{
    Matrix m1(5, 3.0f);
    Matrix m2(10000);
    Matrix m3(10000,10000,10.0f);
    Matrix m4(1000,1000,-11.0f,false);
    Matrix m5(5,false);

    std::cout << "done\n";
}

#endif
