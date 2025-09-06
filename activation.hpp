#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

namespace NNActivation {
	// Sigmoid activation function
	// σ(x) = 1 / (1 + e^-x)
	NNMatrix sigmoid(NNMatrix input) {
		input.forEach([](double* val, int, int) {
			*val = 1.0 / (1.0 + std::exp(-*val));
		});
		return input;
	}
	// Derivative of sigmoid activation function
	// σ'(x) = σ(x) * (1 - σ(x))
	NNMatrix sigmoidDerivative(NNMatrix input) {
		input.forEach([](double* val, int, int) {
			double sigma = 1.0 / (1.0 + std::exp(-*val));
			*val = sigma * (1.0 - sigma);
		});
		return input;
	}
	// ReLU activation function
	// ReLU(x) = max(0, x)
	NNMatrix relu(NNMatrix input) {
		input.forEach([](double* val, int, int) {
			*val = std::max(0.0, *val);
		});
		return input;
	}
	// Derivative of ReLU activation function
	// ReLU'(x) = x if x > 0 else 0
	NNMatrix reluDerivative(NNMatrix input) {
		input.forEach([](double* val, int, int) {
			*val = *val > 0.0 ? 1.0 : 0.0;
		});
		return input;
	}
};

#endif
