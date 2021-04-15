#pragma once
#include <iostream> // overriding cout

#define yeet throw
#define DUMMY_SIZE 16 // 16 bytes of dummy memory per class
#define INDEX(ROW,COL,N_ROWS,N_COLS) ((ROW)*(N_COLS)+(COL)) // the matrices are stored in row-major order

class Matrix
{
    int dim1, dim2;
    float *matrix; // if GPU enabled, this would be stored on the device
    bool gpu_enabled;
    void *dummy; // this is dummy memory that any gpu matrix has to do random stuff on the gpu, since allocating new memory everytime is slow
    // Matrix() {}; // hide default constructor
public:
    /* Constructors and destructors */
    Matrix(){dim1 = 0; dim2 = 0;}
    Matrix(int, bool=true); // random constructor (vector)
    Matrix(int, float, bool=true); // uniform constructor

    Matrix(int, int, bool=true); // random constructor (matrix)
    Matrix(int, int, float, bool=true); // uniform constructor

    Matrix(float*, int, bool=true); // 1d matrix from 1d array
    Matrix(float*, int, int, bool=true); // 2d matrix from 1d array
    Matrix(float**, int, int, bool=true); // 2d matrix from 2d array

    Matrix(const Matrix&); // copy constructor

    ~Matrix(); // destructor

    /* Getters */

    bool is_gpu(void) const { return gpu_enabled; }
    int get_dim1(void) const { return dim1; }
    int get_dim2(void) const { return dim2; }
    float* get_matrix(void) const { return matrix; }


    /* Comparison operators */
    bool operator== (const Matrix&) const;
    bool operator!= (const Matrix&) const;

    /* basic matrix operations */
    Matrix operator+ (const Matrix&) const; // matrix addition
    Matrix operator- (const Matrix&) const; // matrix subtraction

    Matrix operator* (float) const; // scalar multiplication
    Matrix operator/ (float) const; // scalar division

    Matrix operator* (const Matrix&) const; // matrix multiplication


    /* assignment operators */
    void operator+= (const Matrix&);
    void operator-= (const Matrix&);
    void operator*= (float);
    void operator/= (float);

    /* functions */

	void rezero(void); // rezero all elements of the matrix

    Matrix o(const Matrix&) const; // elementwise product

    Matrix T(void) const; // transpose
    void T_inplace(void); // in place transpose
    float dot(const Matrix&) const; // dot product

    void print(int = 0) const; // prints the matrix. default full matrix
    friend std::ostream& operator<<(std::ostream&, const Matrix&); // sets the cout to the default print size

    /* transformation functions */
    void to_gpu(void); // transform the gpu matrix to cpu
    void to_cpu(void); // transform the cpu matrix to gpu

    Matrix get_gpu(void) const; // gets the GPU version
    Matrix get_cpu(void) const; // get the CPU version
};
