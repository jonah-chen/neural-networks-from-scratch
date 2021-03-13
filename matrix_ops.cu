#include "matrix.cuh"

#define BLOCK_SIZE 1024
#define DIVIDE(A,B) (((A)+(B)-1)/(B))
#define BLOCKS(N) DIVIDE(N,BLOCK_SIZE)
#define TILE_DIM 32 // total shared memory usage 32*32 * 2(matrices) * 4(sizeof(float)) = 8kB

///////////////////////////////////////////////////////////////////////////
// Comparison

/* PARELLEL FUNCTION
 * Objective: Compare two matrices and store the answer in bool answer
 * Requirements:
 * 1) The matrices must be the same size.
 * 2) The answer must be a pointer in DEVICE memory
 *
 * Shared Memory Requirements: NONE
 * Runtime: Linear Parellel
 */
__global__ void compare(float* a, float* b, const int size, bool* answer)
{
    if (answer)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < size && a[i] != b[i])
            *answer = false;
    }
}

/* Return true if the matrices are identical to each other and false otherwise */
bool Matrix::operator== (const Matrix& other) const
{
    if (gpu_enabled != other.is_gpu())
        throw "cannot compare gpu matrix with cpu matrix\n";

    if (dim1 != other.get_dim1() || dim2 != other.get_dim2())
        return false;

    float* other_matrix = other.get_matrix();

    if (gpu_enabled)
    {
        bool host_ans;
        cudaMemset(dummy, 1, sizeof(bool)); // set the value of the pointer in device memory to false
        compare <<<BLOCKS(dim1*dim2),BLOCK_SIZE>>> (matrix, other_matrix, dim1*dim2, (bool*)dummy);
        cudaMemcpy(&host_ans, dummy, sizeof(bool), cudaMemcpyDeviceToHost);
        return host_ans;
    }
    else
    {
        for (int i = 0; i < dim1*dim2; ++i)
            if (matrix[i]!=other_matrix[i])
                return false;
        return true;
    }
}

/* Return false if the matrices are identical to each other and true otherwise */
bool Matrix::operator!= (const Matrix& other) const
{
    return !(*this == other);
}

///////////////////////////////////////////////////////////////////////////
// Addition, subtraction, scalar multiplication

/* PARELLEL FUNCTION
 * Objective: Add two matrices and stores it in a third matrix ans.
 * Requirements:
 * 1) The matrices must be the same size.
 * 2) The answer must be a pointer in DEVICE memory and a block sizeof(float)*size must be allocated
 *
 * Shared Memory Requirements: NONE
 * Runtime: Linear Parellel
 */
__global__ void add(float* a, float* b, const int size, float* ans)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
        ans[i] = a[i] + b[i];
}

/* PARELLEL FUNCTION
 * Objective: Subtract two matrices and stores it in a third matrix ans.
 * Requirements:
 * 1) The matrices must be the same size.
 *
 * Shared Memory Requirements: NONE
 * Runtime: Linear Parellel
 */
__global__ void subtract(float* a, float* b, const int size, float* ans)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
        ans[i] = a[i] - b[i];
}

/* PARELLEL FUNCTION
 * Objective: Multiplies a matrix by a scalar and stores it in a third matrix.
 *
 * Shared Memory Requirements: NONE
 * Runtime: Linear Parellel
 */
__global__ void scale(float* a, const float scalar, const int size, float* ans)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
        ans[i] = a[i]*scalar;
}


Matrix Matrix::operator+ (const Matrix& other) const
{
    if (!(gpu_enabled && other.is_gpu()))
        throw "cannot add because one or more of the matrices are not gpu\n";
    if (dim1 != other.get_dim1() || dim2 != other.get_dim2())
        throw "cannot add two matrices of different dimensions\n";

    Matrix ans (dim1, dim2, 0.0f);

    add <<<BLOCKS(dim1*dim2), BLOCK_SIZE>>> (matrix, other.get_matrix(), dim1*dim2, ans.get_matrix());
    return ans;
}

void Matrix::operator+= (const Matrix& other)
{
    if (!(gpu_enabled && other.is_gpu()))
        throw "cannot add because one or more of the matrices are not gpu\n";
    if (dim1 != other.get_dim1() || dim2 != other.get_dim2())
        throw "cannot add two matrices of different dimensions\n";

    add <<<BLOCKS(dim1*dim2), BLOCK_SIZE>>> (matrix, other.get_matrix(), dim1*dim2, matrix);
}


Matrix Matrix::operator- (const Matrix& other) const
{
    if (!(gpu_enabled && other.is_gpu()))
        throw "cannot add because one or more of the matrices are not gpu\n";
    if (dim1 != other.get_dim1() || dim2 != other.get_dim2())
        throw "cannot add two matrices of different dimensions\n";

    Matrix ans (dim1, dim2, 0.0f);

    subtract <<<BLOCKS(dim1*dim2), BLOCK_SIZE>>> (matrix, other.get_matrix(), dim1*dim2, ans.get_matrix());
    return ans;
}

void Matrix::operator-= (const Matrix& other)
{
    if (!(gpu_enabled && other.is_gpu()))
        throw "cannot add because one or more of the matrices are not gpu\n";
    if (dim1 != other.get_dim1() || dim2 != other.get_dim2())
        throw "cannot add two matrices of different dimensions\n";

    subtract <<<BLOCKS(dim1*dim2), BLOCK_SIZE>>> (matrix, other.get_matrix(), dim1*dim2, matrix);
}

Matrix Matrix::operator* (const float scalar) const
{
    if (!gpu_enabled)
        throw "cannot multiply because the matrices is not gpu enabled\n";

    Matrix ans (dim1, dim2, 0.0f);

    scale <<<BLOCKS(dim1*dim2), BLOCK_SIZE>>> (matrix, scalar, dim1*dim2, ans.get_matrix());
    return ans;
}

void Matrix::operator*= (const float scalar)
{
    if (!gpu_enabled)
        throw "cannot multiply because the matrices is not gpu enabled\n";

    scale <<<BLOCKS(dim1*dim2), BLOCK_SIZE>>> (matrix, scalar, dim1*dim2, matrix);
}

Matrix Matrix::operator/ (const float scalar) const
{
    if (scalar)
        return *this * (1.0f / scalar);
    else
        throw "You're tryin to divide by zero, idiot!\n";
}

void Matrix::operator/= (const float scalar)
{
    if (scalar)
        *this *= (1.0f / scalar);
    else
        throw "You're tryin to divide by zero, idiot!\n";
}

//////////////////////////////////////////////////////////////////////////
// Matrix multiplication


/* PARELLEL FUNCTION
 * Objective: Does matrix multiplication between two matrices and stores it in a third. ans = ab.
 * Requirements:
 * 1) The matrices dimensions must match, including the ans matrix.
 *
 * Shared Memory Requirements: 8kb
 * Runtime: O(a_r * a_c__b_r * b_c / THREADS)
 */

__global__ void matmul(float* A, float* B, float* C, int ARows, int ACols, int BRows,
                       int BCols, int CRows, int CCols)
{
    float CValue = 0.0f;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

#pragma unroll
    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

        if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
            As[threadIdx.y][threadIdx.x] = A[INDEX(Row,k*TILE_DIM+threadIdx.x,ARows,ACols)];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
            Bs[threadIdx.y][threadIdx.x] = B[INDEX(k*TILE_DIM+threadIdx.y,Col,BRows,BCols)];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int n = 0; n < TILE_DIM; ++n)
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

        __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[INDEX(blockIdx.y * blockDim.y + threadIdx.y, (blockIdx.x * blockDim.x)+ threadIdx.x, CRows,CCols)] = CValue;
}


Matrix Matrix::operator* (const Matrix& other) const
{
    if (!(gpu_enabled && other.is_gpu()))
        throw "cannot multiply because one or more of the matrices are not gpu\n";

    if (dim2 != other.get_dim1())
        throw "matrix dimensions are not compatiable.";

    Matrix ans(dim1, other.get_dim2(), 0.0f);

    dim3 grid (DIVIDE(dim1,TILE_DIM), DIVIDE(dim2,TILE_DIM), 1);
    dim3 block (TILE_DIM,TILE_DIM,1);

    matmul<<<grid, block, sizeof(float)*TILE_DIM*TILE_DIM>>> (matrix, other.get_matrix(), ans.get_matrix(), dim1, dim2, other.get_dim1(), other.get_dim2(), ans.get_dim1(), ans.get_dim2());

    return ans;
}


__global__ void had(float* a, float* b, const int size, float* ans)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)
		ans[i] = a[i] * b[i];
}

Matrix Matrix::o(const Matrix& other) const
{
	if (!(gpu_enabled && other.is_gpu()))
		throw "cannot hadamard because one or more of the matrices are not gpu\n";
	if (dim1 != other.get_dim1() || dim2 != other.get_dim2())
		throw "cannot perform hadamard product on matrices of different dimensions";

	Matrix ans(dim1, dim2, 0.0f);
	had <<<BLOCKS(dim1*dim2), BLOCK_SIZE>>> (matrix, other.get_matrix(), dim1*dim2, ans.get_matrix());

	return ans;
}

__global__ void transpose (float* a, float* ans, const int dim1, const int dim2)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < dim1 && y < dim2)
		ans[INDEX(y,x,dim2,dim1)] = a[INDEX(x,y,dim1,dim2)];
}

__global__ void transpose (float* a, const int dim)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	float tmp;
	if (x < dim && y < dim && x < y)
	{
		tmp = a[INDEX(x,y,dim,dim)];
		a[INDEX(x,y,dim,dim)] = a[INDEX(y,x,dim,dim)];
		a[INDEX(y,x,dim,dim)] = tmp;
	}
}

Matrix Matrix::T(void) const
{
	if (!gpu_enabled)
		throw "cannot transpose a non-gpu matrix";
	Matrix ans(dim2, dim1, 0.0f);
	transpose<<<BLOCKS(dim1*dim2),BLOCK_SIZE>>>(matrix, ans.get_matrix(), dim1, dim2);

	return ans;
}

void Matrix::T_inplace(void)
{
	if (dim1 != dim2)
		*this = this->T();
	else
		transpose <<<BLOCKS(dim1*dim2),BLOCK_SIZE>>> (matrix, dim1);
}


float Matrix::dot(const Matrix& other) const
{
	if ((dim1 > 1 && dim2 > 1) || (other.get_dim1() > 1) && (other.get_dim2() > 1))
		throw "one cannot take the dot product of two matrices\n";
	if (dim1 + dim2 != other.get_dim2() + other.get_dim1())
		throw "the dimensions don't match.\n";

	float* partial_ans;
	cudaMalloc((void**)&partial_ans, sizeof(float)*(dim1+dim2-2));
	int reqs = dim1+dim2 -1;
	throw "the dimensions don't match.\n";

}
