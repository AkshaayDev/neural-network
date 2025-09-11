#ifndef LOSS_HPP
#define LOSS_HPP

#include "neural_network.hpp"

namespace NNLoss {
	// Mean squared error
	// MSE = 1/n * ∑(ŷ_i - y_i)^2
	inline double MSE(NNMatrix predicted, NNMatrix real) {
		double result = 0;
		predicted.forEach([real, &result](double *val, int i, int j) {
			result += std::pow(*val - real.data[i][j], 2);
		});
		return result / real.rows();
	}
	// Derivative of mean squared error
	// MSE' = 2/n * (ŷ_i - y_i)
	inline NNMatrix MSEDerivative(NNMatrix predicted, NNMatrix real) {
		predicted.forEach([&real](double *val, int i, int j) {
			*val = 2.0 / real.rows() * (*val - real.data[i][j]);
		});
		return predicted;
	}
}

#endif
