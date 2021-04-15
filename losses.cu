#include "losses.cuh"

#define BLOCK_SIZE 1024
#define DIVIDE(A,B) ((A+B-1)/B)
#define BLOCKS(N) DIVIDE(N,BLOCK_SIZE)

__global__ void gpu_deriv_mse(float* yt, float* yp, const unsigned int size, float* ans)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
		ans[i] = yp[i] - yt[i];
}

float losses::func_mse(const Matrix& y_true, const Matrix& y_pred)
{
	if (y_true.is_gpu() != y_pred.is_gpu())
		yeet "One of the two matrices is GPU enabled, and the other isn't";

	if (y_true.get_dim1() != y_pred.get_dim1() or y_true.get_dim2() != y_pred.get_dim2())
		yeet "The dimensions does not match";

	// there isn't much of a point doing it on the GPU rn, so we'll do it on the CPU
	float loss = 0.0f, *yt, *yp;

	if (y_true.is_gpu())
	{
		yt = y_true.get_cpu().get_matrix();
		yp = y_pred.get_cpu().get_matrix();
	}
	else
	{
		yt = y_true.get_matrix();
		yp = y_pred.get_matrix();
	}

	for (int i = 0; i < y_true.get_dim1()*y_true.get_dim2(); ++i)
		loss += (yt[i] - yp[i]) * (yt[i] - yp[i]);

	return loss * 0.5f;
}

Matrix losses::deriv_mse(const Matrix& y_true, const Matrix& y_pred)
{
	if (y_true.is_gpu() != y_pred.is_gpu())
		yeet "One of the two matrices is GPU enabled, and the other isn't";

	if (y_true.get_dim1() != y_pred.get_dim1() or y_true.get_dim2() != y_pred.get_dim2())
		yeet "The dimensions does not match";

	if (y_true.is_gpu())
	{
		Matrix ans(y_true.get_dim1(), y_true.get_dim2(), 0.0f);
		gpu_deriv_mse <<<BLOCKS(y_true.get_dim1()*y_true.get_dim2()), BLOCK_SIZE>>>(y_true.get_matrix(), y_pred.get_matrix(), y_true.get_dim2()*y_true.get_dim1(), ans.get_matrix());
		return ans;
	}
	else
	{
		Matrix ans(y_true.get_dim1(), y_true.get_dim2(), 0.0f, false);
		float *yt = y_true.get_matrix(), *yp = y_pred.get_matrix(), *a = ans.get_matrix();
		for (unsigned int i = 0; i < y_true.get_dim1()*y_true.get_dim2(); ++i)
			a[i] = yp[i] - yt[i];
		return ans;
	}
}