//
// Created by hina on 2021-03-13.
//

#include "activation.cuh"

__device__ float activation::func_relu(float input_num)
{
	return input_num > 0 ? input_num : 0.0f;
}

__device__ float activation::deriv_relu(float input_num)
{
	return input_num > 0 ? 1.0f : 0.0f;
}