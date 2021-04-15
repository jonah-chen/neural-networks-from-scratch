#pragma once

#include "layer.cuh"
#include "dense.cuh"
#include "losses.cuh"

#include <iterator>
#include <list>
#include <vector>

class Model
{
	std::list<Layer*> layers;
	Loss_Function loss_function;


public:
	float train_batch(std::vector<Matrix>::iterator, std::vector<Matrix>::iterator, const unsigned int, const float, const float=0.0f); // train one batch and return the average loss on that batch

	Model(Loss_Function);

	void operator+= (Layer*); // add a new layer to the neural network
	Matrix feed (const Matrix&) const; // use the model to predict from a given input

	void train(const std::vector<Matrix>&, const std::vector<Matrix>&, const unsigned int batch_size);
};