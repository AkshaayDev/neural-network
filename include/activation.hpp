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
};

#endif
