#pragma once

#include "layer.cuh"
#include "activation.cuh"

class Dense : public Layer
{
	Matrix weight, weight_delta; // in x out
	Matrix bias, bias_delta; // out x 1

protected:
	/* activation function that will be applied.
	 * MUST BE __device__ FUNCTION
	 * Limited to functions that does not depend on the other weights.
	 */
	Activation_Function activation_func;


public:
	// neurons = activation_func(weights * prev_layer_neuron + bias)

	Dense(int, int, Activation_Function=activation::LINEAR); // input and output dimensions, activation function defaults to NULL

	void feedforward_update(const Matrix &prev_neurons) override;
	Matrix feedforward(const Matrix &prev_neurons) const override;
	Matrix backpropogate(const Matrix &old_grad) override;

	void backpropogate_update(const Matrix &old_grad) override;

	void gradient_update(float learning_rate, float momentum);
};
