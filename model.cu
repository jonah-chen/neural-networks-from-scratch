#include "model.cuh"

float Model::train_batch(std::vector<Matrix>::iterator X_i, std::vector<Matrix>::iterator Y_i, const unsigned int batch_size, const float lr, const float momentum)
{
	float loss = 0.0f;
	for (unsigned int i = 0; i < batch_size; i++)
	{
		// create the inputs
		Matrix tmp = *X_i;

		// feedforward through the neural network
		for (auto layer = layers.begin(); layer != layers.end(); ++layer)
		{
			(*layer)->feedforward_update(tmp);
			tmp = (*layer)->neurons;
		}

		// calculate the loss
		loss += loss_function.func(*Y_i, tmp);

		// calculate the derivative for the loss function
		// reuse temporary variable, as it is no longer needed
		tmp = loss_function.deriv(*Y_i, tmp);

		// now backpropogate through the layers
		for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer)
			tmp = (*layer)->backpropogate(tmp);

		++X_i; ++Y_i; // advance the iterators
	}

	for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer)
		(*layer)->gradient_update(lr, momentum);

	return loss / (float) batch_size;
}

Model::Model(Loss_Function loss_function)
{
	this->loss_function = loss_function;
}

void Model::operator+=(Layer* layer)
{
	//if (layers.empty() or layers.back()->get_out_shape() == layer.get_in_shape())
		layers.push_back(layer);
	//else
		//yeet "This layer's input shape is not compatible";
}

Matrix Model::feed (const Matrix& input) const
{
	Matrix tmp = (*(layers.begin()))->feedforward(input);

	// feedforward through the neural network
	auto layer = layers.begin();
	for (++layer; layer != layers.end(); ++layer)
		tmp = (*layer)->feedforward(tmp);
	return tmp;
}