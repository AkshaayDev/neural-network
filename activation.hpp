#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "./neural-network.hpp"

namespace NNActivation {
	// Sigmoid activation function
	// σ(x) = 1 / (1 + e^-x)
	inline NNMatrix sigmoid(NNMatrix input) {
		input.forEach([](double* val, int, int) {
			*val = 1.0 / (1.0 + std::exp(-*val));
		});
		return input;
	}
	// Derivative of sigmoid activation function
	// σ'(x) = σ(x) * (1 - σ(x))
	inline NNMatrix sigmoidDerivative(NNMatrix input) {
		input.forEach([](double* val, int, int) {
			double sigma = 1.0 / (1.0 + std::exp(-*val));
			*val = sigma * (1.0 - sigma);
		});
		return input;
	}
	// ReLU activation function
	// ReLU(x) = max(0, x)
	inline NNMatrix relu(NNMatrix input) {
		input.forEach([](double* val, int, int) {
			*val = std::max(0.0, *val);
		});
		return input;
	}
	// Derivative of ReLU activation function
	// ReLU'(x) = x if x > 0 else 0
	inline NNMatrix reluDerivative(NNMatrix input) {
		input.forEach([](double* val, int, int) {
			*val = *val > 0.0 ? 1.0 : 0.0;
		});
		return input;
	}
	// Hyperbolic tangent activation function
	// tanh(x) = (e^x-e^-x)/(e^x+e^-x)
	inline NNMatrix tanh(NNMatrix input) {
		input.forEach([](double* val, int, int) {
			*val = std::tanh(*val);
		});
		return input;
	}
	// Derivative of hyperbolic tangent activation function
	// tanh'(x) = 1 - tanh(x)^2
	inline NNMatrix tanhDerivative(NNMatrix input) {
		input.forEach([](double* val, int, int) {
			*val = 1 - std::pow(std::tanh(*val), 2);
		});
		return input;
	}
	// Softmax activation function (output only)
	// softmax(Z)_i = e^(z_i) / sum_j=1^N e^(z_j)
	inline NNMatrix softmax(NNMatrix input) {
		double sum = 0;
		double max = input.max();
		input.forEach([&sum, max, &input](double* val, int, int) {
			*val = std::exp(*val - max); // Subtract max for numerical stability while maintaining output
			sum += *val;
		});
		return input / sum;
	}
	// There is no softmax activation function derivative
	// Each preactivation affects every activation so the derivative would be a Jacobian matrix(n x n)
	// Instead, the NeuralNetwork class handles its derivative for cross entropy loss
}

// Activation functions are network attributesy and need to be specified in the network
// The strings in this namespace are used to identify the activation functions of the network for saving and loading
namespace NNActivationType {
	const std::string Sigmoid = "sigmoid";
	const std::string ReLU = "relu";
	const std::string Tanh = "tanh";
	const std::string Softmax = "softmax";
}

#endif
