#pragma once

/* This is a list of the activation functions
 * The functions in this namespace can be used to construct activation layers
 */
#include "layer.cuh"

struct Activation_Function
{
	float(*func)(float); // the activation function
	float(*deriv)(float); // the derivative of the activation function
};

namespace activation
{
	// aux functions
	__device__ float func_relu(float input_num);
	__device__ float deriv_relu(float input_num);

	// functions to be used
	const Activation_Function LINEAR = (Activation_Function){nullptr, nullptr}, // linear is default activation function
	RELU = (Activation_Function){func_relu, deriv_relu}; // relu activation function
}
