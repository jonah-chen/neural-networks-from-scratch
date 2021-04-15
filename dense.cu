#include "dense.cuh"

#define BLOCK_SIZE 1024
#define DIVIDE(A,B) ((A+B-1)/B)
#define BLOCKS(N) DIVIDE(N,BLOCK_SIZE)

/* PARALLEL FUNCTION:
 * Objective: Applies the activation function to a set of neurons
 * Runtime: O(n) parallel
 * Arguments:
 * 		matrix: the neurons the activation function to apply to
 * 		size: the number of neurons in matrix
 * 		activation_func: the activation function
 * 	Requirements: activation_func MUST be a __device__ function
 *	Exceptions: None. The case where activation_func is not a __device__ function is UNHANDLED.
 */
__global__ void activate(float* matrix, unsigned int size, float(*activation_func)(float))
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size)
		matrix[i] = activation_func(matrix[i]);
}

/* PARALLEL FUNCTION
 * Objective: Apply the derivative of the activation function using chain rule to the gradient of the layer
 * Runtime: O(n) parallel
 *
 */
__global__ void apply_activation_deriv(float* grads, float* outputs, unsigned int size, float(*activation_deriv)(float))
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size)
		grads[i] *= activation_deriv(outputs[i]);
}

Dense::Dense(int input_shape, int output_shape, Activation_Function activation_func) : Layer(input_shape, output_shape)
{
	this->activation_func = activation_func;
	bias = Matrix(output_shape, 0.0f);
	bias_delta = Matrix(output_shape, 0.0f);
	weight = Matrix(output_shape, input_shape);
	weight_delta = Matrix(output_shape, input_shape, 0.0f);
	p_neurons = Matrix(input_shape, 0.0f);
	neurons = Matrix(output_shape, 0.0f);
	stored_grad = Matrix(input_shape, 0.0f);
}

/* Objective: Update the neurons of this layer by feeding forward the neurons of the previous layer
 * Runtime: O(n^1.5) average
 * Arguments:
 * 		prev_neurons (Matrix const reference), the matrix of the neuron of the previous layer
 * Requirements: prev_neurons MUST be GPU enabled matrix
 * 				 The dimensions MUST match
 * Exceptions:
 * 		GPU is not enabled for previous neurons
 * 		The dimensions of the matrices does not matched
 */

void Dense::feedforward_update(const Matrix& prev_neurons)
{
	if (!prev_neurons.is_gpu())
		yeet "GPU is not enabled for previous neurons";

	if (prev_neurons.get_dim1() != input_shape)
		yeet "The dimensions does not match";

	p_neurons = prev_neurons;
	neurons = weight * prev_neurons + bias;

	if (activation_func.func) // if the activation function exists, apply the activation function.
		activate<<<BLOCKS(output_shape), BLOCK_SIZE>>>(neurons.get_matrix(), output_shape, activation_func.func);
}

/* Objective: Feedforward the neurons of the previous layer using the weights and biases of this layer and returning the output
 * Runtime: O(n^1.5) average
 * Arguments:
 * 		prev_neurons (Matrix const reference), the matrix of the neuron of the previous layer
 * Requirements: prev_neurons MUST be GPU enabled matrix
 * 				 The dimensions MUST match
 * Exceptions:
 * 		GPU is not enabled for previous neurons
 * 		The dimensions of the matrices does not matched
 */

Matrix Dense::feedforward(const Matrix &prev_neurons) const
{
	if (!prev_neurons.is_gpu())
		yeet "GPU is not enabled for previous neurons";

	if (prev_neurons.get_dim1() != input_shape)
		yeet "The dimensions does not match";

	Matrix ans (output_shape, 0.0f);

	ans = weight * prev_neurons + bias;

	if (activation_func.func) // if the activation function exists, apply activation function
		activate<<<BLOCKS(output_shape), BLOCK_SIZE>>>(neurons.get_matrix(), output_shape, activation_func.func);

	return ans;
}

/* Objective: Using the previous gradient (from the subsequent layer), calculate the gradient of this layer and store it in stored_grad.
 * 			  Update the values of the weight and bias delta based on this gradient.
 * Arguments:
 * 		old_grad (Matrix const reference): the gradient of the subsequent layer
 * Requirements:
 * 		old_grad MUST be GPU enabled matrix
 * 		The dimensions of the matrices must match old_grad should be (output_shape x 1)
 */

Matrix Dense::backpropogate(const Matrix &old_grad)
{
	if (!old_grad.is_gpu())
		yeet "GPU is not enabled for previous neurons";

	if (old_grad.get_dim1() != output_shape)
		yeet "The dimensions does not match";

	// first use the derivative of the activation function
	Matrix old_grad_cpy = old_grad;
	apply_activation_deriv<<<BLOCKS(output_shape), BLOCK_SIZE>>>(old_grad_cpy.get_matrix(), neurons.get_matrix(), output_shape, activation_func.deriv);

	// update the weight and bias deltas
	bias_delta += old_grad_cpy;
	weight_delta += old_grad_cpy * p_neurons.T();

	return weight.T() * old_grad_cpy;
}

void Dense::backpropogate_update(const Matrix &old_grad)
{
	stored_grad = backpropogate(old_grad);
}

/* Objective: Update the weights and biases with the calculated weight and bias deltas and the given learning rate.
 * Arguments:
 * 		learning rate: the learning rate of the neural network
 * 		momentum: the batch momentum after gradient descent. defaults to 0.0 meaning the deltas are rezeroed. Momentum must be between 0 and 1.
 * Runtime: O(n) parallel
 * Exceptions: None
 */

void Dense::gradient_update(float learning_rate, float momentum)
{
	if (momentum > 1.0f)
		yeet "Momentum cannot be greater than one";

	if (momentum < 0.0f)
		yeet "Momentum cannot be negative";

	weight += weight_delta * learning_rate;
	bias += bias_delta * learning_rate;

	if (momentum == 0.0f)
	{
		weight *= momentum;
		bias *= momentum;
	}
	else
	{
		weight.rezero();
		bias.rezero();
	}
}
