#include "matrix.cuh"

#define PRINT_MAX 10
#define TRUNC(A) ((A) < PRINT_MAX ? (A) : PRINT_MAX)

void Matrix::print(int sz) const // prints the entire matrix
{
    if (sz)
    {

    }
    else
    {
        // print everything
        if (gpu_enabled)
        {
            float* a = new float[dim1*dim2];
            cudaMemcpy(a, matrix, sizeof(float)*dim1*dim2, cudaMemcpyDeviceToHost);
            for (int i = 0; i < dim1; ++i)
            {
                for (int j = 0; j < dim2; ++j)
                    std::cout << a[INDEX(i,j,dim1,dim2)] << ",";

                std::cout << "\n";
            }
            delete[] a;
        }
        else
        {
            for (int i = 0; i < dim1; ++i)
            {
                for (int j = 0; j < dim2; ++j)
                    std::cout << matrix[INDEX(i,j,dim1,dim2)] << ",";

                std::cout << "\n";
            }
        }
    }
}
std::ostream& operator<<(std::ostream& os, const Matrix& mat) // sets the cout to the default print size
{
    os << "MATRIX " << (mat.is_gpu() ? "(GPU): (" : "(CPU): (");
    os << mat.get_dim1() << " x " << mat.get_dim2() << ")\n";

    int d1 = mat.get_dim1(), d2 = mat.get_dim2();

    if (mat.is_gpu()) // gpu version
    {
        // copy every line of the matrix
        if (mat.get_dim1() * mat.get_dim2() < 4096) // if there are less than 1024 elements, then just copy the cpu version
        {
            float *a = new float[d1*d2];
            cudaMemcpy(a, mat.get_matrix(), sizeof(float)*d1*d2, cudaMemcpyDeviceToHost);

            if (d2 == 1)
            {
                os << "[";
                for (int i = 0; i < TRUNC(d1); ++i)
                    os << a[i] << ",";
                os << "...]\n";
            }
            else
            {
                for (int i = 0; i < TRUNC(d1); ++i)
                {
                    for (int j = 0; j < TRUNC(d2); ++j)
                        os << a[INDEX(i,j,d1,d2)] << ",";
                    os << "...\n";
                }
                os << "......\n";
            }
            delete[] a;
        }
        else
        {
            // matrix is too big. too slow to copy the entire thing. we just copy the 100 numbers we want
            float a[PRINT_MAX];

            if (d2 == 1)
            {
                cudaMemcpy(a, mat.get_matrix(), sizeof(float)*PRINT_MAX, cudaMemcpyDeviceToHost);
                os << "[";
                for (int i = 0; i < TRUNC(d1); ++i)
                    os << a[i] << ",";
                os << "...]\n";
            }
            else
            {
                for (int i = 0; i < TRUNC(d1); ++i)
                {
                    // copy the memory of only one of the lines (of 10 things)
                    cudaMemcpy(a, mat.get_matrix() + INDEX(i,0,d1,d2), sizeof(float)*PRINT_MAX, cudaMemcpyDeviceToHost);
                    // print the stuff
                    for (int j = 0; j < TRUNC(d2); ++j)
                        os << a[j] << ",";

                    os << "...\n";
                }
                os << "......\n";
            }
        }
    }
    else // cpu version
    {
        float *a = mat.get_matrix();

        if (d2 == 1) // if it's just a column matrix, we can save some space on the terminal by putting it as a row. We will use square brackets to denote this
        {
            os << "[";
            for (int i = 0; i < TRUNC(d1); ++i)
                os << a[i] << ",";
            os << "...]\n";
        }
        else
        {
            for (int i = 0; i < TRUNC(d1); ++i)
            {
                for (int j = 0; j < TRUNC(d2); ++j)
				{
					os << a[INDEX(i, j, d1, d2)] << ",";
				}
                os << "...\n";
            }
            os << "......\n";
        }
    }
    return os;
}
