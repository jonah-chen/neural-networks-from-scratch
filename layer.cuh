#pragma once

#include "matrix.cuh"

/* ABSTRACT TEMPLATE CLASS
 * The Layer class serves as a template for the other layers of the neural network.
 */
class Layer
{
protected:
	int input_shape;
	int output_shape;
	Layer(int, int);

public:
	Matrix neurons, p_neurons; // this will be the neurons of the layer
	Matrix stored_grad;


	/* Getters */
	int get_in_shape() const {return input_shape;}
	int get_out_shape() const {return output_shape;}

	void gradient_update(float, float=0.0) {} // update to weights and biases. some layers may not have weights or biases, so it does nothing by default

	/* THESE FUNCTIONS ARE TO BE IMPLEMENTED FOR EVERY LAYER */

	virtual Matrix feedforward(const Matrix&) const {} // feedforwards and returns a new matrix
	virtual void feedforward_update(const Matrix&) {} // feedforwards into the neurons matrix
	virtual Matrix backpropogate(const Matrix&) {} // returns the gradient when backpropogating through the matrix the original matrix
	virtual void backpropogate_update(const Matrix&) {} // modifies the previous gradients when it backpropgates through this layer
};