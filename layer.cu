#include "layer.cuh"

Layer::Layer(int input_shape, int output_shape)
{
	this->input_shape = input_shape;
	this->output_shape = output_shape;
}