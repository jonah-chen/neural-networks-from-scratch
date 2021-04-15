#pragma once

#ifndef LOSS
#define LOSS

#include "matrix.cuh"

struct Loss_Function
{
	float(*func)(const Matrix&, const Matrix&);
	Matrix(*deriv)(const Matrix&, const Matrix&);
};

namespace losses
{
	float func_mse(const Matrix& y_true, const Matrix& y_pred);
	Matrix deriv_mse(const Matrix& y_true, const Matrix& y_pred);

	const Loss_Function MSE = (Loss_Function) {func_mse, deriv_mse};
}

#endif