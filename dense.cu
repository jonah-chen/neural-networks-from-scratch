#include "dense.cuh"

Dense::Dense(int input_shape, int output_shape)
{
	this->input_shape = input_shape;
	this->output_shape = output_shape;
	bias = Matrix(output_shape);
	weight = Matrix(input_shape, output_shape)
}