#include "matrix.cuh"

void Matrix::to_gpu(void)
{
	if (!gpu_enabled)
	{
		gpu_enabled = true;
		float* d_matrix;
		if (cudaMalloc((void**)&d_matrix, sizeof(float)*dim1*dim2) != cudaSuccess)
			throw "memory allocation failed\n";
		cudaMemcpy(d_matrix, matrix, sizeof(float)*dim1*dim2, cudaMemcpyHostToDevice);
		delete[] matrix;
		matrix = d_matrix;
	}
}

void Matrix::to_cpu(void)
{
	if (gpu_enabled)
	{
		gpu_enabled = false;
		float *h_matrix = new float[dim1*dim2];
		cudaMemcpy(h_matrix, matrix, sizeof(float)*dim1*dim2, cudaMemcpyDeviceToHost);
		cudaFree(matrix);
		matrix = h_matrix;
	}
}

Matrix Matrix::get_gpu(void) const
{
	if (gpu_enabled)
		throw "cannot get another gpu matrix if it is already gpu";
	return Matrix(matrix, dim1, dim2);
}

Matrix Matrix::get_cpu(void) const
{
	if (gpu_enabled)
	{
		float* h_matrix = new float[dim1*dim2];
		cudaMemcpy(h_matrix, matrix, sizeof(float)*dim1*dim2, cudaMemcpyDeviceToHost);
		return Matrix(h_matrix, dim1, dim2, false);
	}
	return Matrix(matrix, dim1, dim2, false);
}