#ifndef LOSS_HPP
#define LOSS_HPP

#include "./neural-network.hpp"

namespace NNLoss {
	// Mean Squared Error
	// MSE = 1/n * ∑(ŷ_i - y_i)^2
	inline double MSE(NNMatrix predicted, NNMatrix real) {
		double sum = 0;
		predicted.forEach([&real, &sum](double *val, int i, int j) {
			sum += std::pow(*val - real[i][j], 2);
		});
		return sum / real.rows();
	}
	// Derivative of Mean Squared Error
	// MSE' = 2/n * (ŷ_i - y_i)
	inline NNMatrix MSEDerivative(NNMatrix predicted, NNMatrix real) {
		predicted.forEach([&real](double *val, int i, int j) {
			*val = 2.0 / real.rows() * (*val - real[i][j]);
		});
		return predicted;
	}
	// Categorical Cross Entropy Loss
	// CCE = - ∑ y_i log(ŷ_i + ε)
	inline double CCE(NNMatrix predicted, NNMatrix real) {
		double sum = 0;
		const double epsilon = 1e-12; // avoid log(0)
		predicted.forEach([&sum, &real, epsilon](double *val, int i, int j) {
			sum -= real[i][j] * std::log(*val + epsilon);
		});
		return sum;
	}
	// Derivative of Categorical Cross Entropy Loss
	// CCE' = - y_i / (ŷ_i + ε)
	inline NNMatrix CCEDerivative(NNMatrix predicted, NNMatrix real) {
		const double epsilon = 1e-12; // avoid / 0
		predicted.forEach([&real, epsilon](double *val, int i, int j) {
			*val = - real[i][j] / (*val + epsilon);
		});
		return predicted;
	}
}

// Loss functions are network properties and need to be specified in the network
// The strings in this namespace are used to identify the loss function in the network for saving and loading
namespace NNLossType {
	const std::string MSE = "mse";
	const std::string CCE = "cce";
}

#endif
