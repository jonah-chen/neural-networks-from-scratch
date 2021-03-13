
#ifndef NEURAL_NETWORKS_FROM_SCRATCH_DENSE_CUH
#define NEURAL_NETWORKS_FROM_SCRATCH_DENSE_CUH

#include "matrix.cuh"

class Dense
{
	int input_shape;
	int output_shape;
	Matrix weight; // in x out
	Matrix bias; // out x 1
public:
	Dense(int, int); // input and output dimensions

	Matrix feedforward(const Matrix&) const;
	void feedforward(const Matrix&, float*) const;

	void gradient_update(const Matrix&, const Matrix&);
};


#endif //NEURAL_NETWORKS_FROM_SCRATCH_DENSE_CUH
