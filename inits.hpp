#ifndef INITS_HPP
#define INITS_HPP

#include "./neural-network.hpp"

namespace NNInitialisations {
	// Weight initialisation functions

	// Uniform Xavier initialisation
	// Initialise weights uniformly across +- sqrt(6/(n_in + n_out))
	inline void xavierUniform(NeuralNetwork& nn) {
		std::mt19937 gen(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
		for (int i = 0; i < nn.weights.size(); i++) {
			double limit = std::sqrt(6.0 / (nn.layers[i] + nn.layers[i + 1]));
			std::uniform_real_distribution<double> dis(-limit, limit);
			nn.weights[i].forEach([&dis, &gen](double *val, int, int) {
				*val = dis(gen);
			});
		}
		nn.epochsTrained = 0;
	}
	// Normal Xavier initialisation
	// Initialise weights with a normal distribution with mean of 0 and standard deviation of sqrt(2/(n_in + n_out))
	inline void xavierNormal(NeuralNetwork& nn) {
		std::mt19937 gen(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
		for (int i = 0; i < nn.weights.size(); i++) {
			double stddev = std::sqrt(2.0 / (nn.layers[i] + nn.layers[i + 1]));
			std::normal_distribution<double> dis(0.0, stddev);
			nn.weights[i].forEach([&dis, &gen](double *val, int, int) {
				*val = dis(gen);
			});
		}
		nn.epochsTrained = 0;
	}
	// Uniform He initialisation
	// Initialise weights uniformly across +- sqrt(6/n_in)
	inline void heUniform(NeuralNetwork& nn) {
		std::mt19937 gen(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
		for (int i = 0; i < nn.weights.size(); i++) {
			double limit = std::sqrt(6.0 / nn.layers[i]);
			std::uniform_real_distribution<double> dis(-limit, limit);
			nn.weights[i].forEach([&dis, &gen](double *val, int, int) {
				*val = dis(gen);
			});
		}
		nn.epochsTrained = 0;
	}
	// Normal He initialisation
	// Initialise weights with a normal distribution with mean of 0 and standard deviation of sqrt(2/n_in)
	inline void heNormal(NeuralNetwork& nn) {
		std::mt19937 gen(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
		for (int i = 0; i < nn.weights.size(); i++) {
			double stddev = std::sqrt(2.0 / nn.layers[i]);
			std::normal_distribution<double> dis(0.0, stddev);
			nn.weights[i].forEach([&dis, &gen](double *val, int, int) {
				*val = dis(gen);
			});
		}
		nn.epochsTrained = 0;
	}

	// Bias initialisation functions

	// Initialise biases to a constant
	inline void constantBias(NeuralNetwork& nn, double constant) {
		for (int i = 0; i < nn.biases.size(); i++) {
			nn.biases[i].fill(constant);
		}
		nn.epochsTrained = 0;
	}
	// Initialise biases to 0
	inline void zeroBias(NeuralNetwork& nn) {
		constantBias(nn, 0.0);
	}
}

// Initialisers are standalone functions and not network properties
// Therefore, the initialisation functions being used need not be specified in the network

#endif
