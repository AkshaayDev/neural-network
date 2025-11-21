#ifndef INITS_HPP
#define INITS_HPP

#include "./neural-network.hpp"

namespace NNInitialization {
	// Weight initialization functions

	// Uniform Xavier initialization
	// Initialize weights uniformly across +- sqrt(6/(in + out))
	inline void xavierUniform(NeuralNetwork& nn) {
		std::mt19937 gen(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
		for (int i = 0; i < nn.depth; i++) {
			DenseLayer* layer = dynamic_cast<DenseLayer*>(nn.layers[i].get());
			if (layer == nullptr) continue; // Continue for non-DenseLayers

			double limit = std::sqrt(6.0 / (layer->inCount + layer->outCount));
			std::uniform_real_distribution<double> dis(-limit, limit);
			layer->W.forEach([&dis, &gen](double *val, int, int) {
				*val = dis(gen);
			});
		}
		nn.iterationsTrained = nn.epochsTrained = 0;
	}
	// Normal Xavier initialization
	// Initialize weights with a normal distribution with mean of 0 and standard deviation of sqrt(2/(in + out))
	inline void xavierNormal(NeuralNetwork& nn) {
		std::mt19937 gen(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
		for (int i = 0; i < nn.depth; i++) {
			DenseLayer* layer = dynamic_cast<DenseLayer*>(nn.layers[i].get());
			if (layer == nullptr) continue; // Continue for non-DenseLayers

			double stddev = std::sqrt(2.0 / (layer->inCount + layer->outCount));
			std::normal_distribution<double> dis(0.0, stddev);
			layer->W.forEach([&dis, &gen](double *val, int, int) {
				*val = dis(gen);
			});
		}
		nn.iterationsTrained = nn.epochsTrained = 0;
	}
	// Uniform He initialization
	// Initialize weights uniformly across +- sqrt(6/in)
	inline void heUniform(NeuralNetwork& nn) {
		std::mt19937 gen(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
		for (int i = 0; i < nn.depth; i++) {
			DenseLayer* layer = dynamic_cast<DenseLayer*>(nn.layers[i].get());
			if (layer == nullptr) continue; // Continue for non-DenseLayers

			double limit = std::sqrt(6.0 / layer->inCount);
			std::uniform_real_distribution<double> dis(-limit, limit);
			layer->W.forEach([&dis, &gen](double *val, int, int) {
				*val = dis(gen);
			});
		}
		nn.iterationsTrained = nn.epochsTrained = 0;
	}
	// Normal He initialization
	// Initialize weights with a normal distribution with mean of 0 and standard deviation of sqrt(2/in)
	inline void heNormal(NeuralNetwork& nn) {
		std::mt19937 gen(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
		for (int i = 0; i < nn.depth; i++) {
			DenseLayer* layer = dynamic_cast<DenseLayer*>(nn.layers[i].get());
			if (layer == nullptr) continue; // Continue for non-DenseLayers

			double stddev = std::sqrt(2.0 / layer->inCount);
			std::normal_distribution<double> dis(0.0, stddev);
			layer->W.forEach([&dis, &gen](double *val, int, int) {
				*val = dis(gen);
			});
		}
		nn.iterationsTrained = nn.epochsTrained = 0;
	}

	// Bias initialization functions

	// Initialize biases to a constant
	inline void constantBias(NeuralNetwork& nn, double constant) {
		for (int i = 0; i < nn.depth; i++) {
			DenseLayer* layer = dynamic_cast<DenseLayer*>(nn.layers[i].get());
			if (layer == nullptr) continue; // Continue for non-DenseLayers

			layer->B.forEach([constant](double *val, int, int) {
				*val = constant;
			});
		}
		nn.iterationsTrained = nn.epochsTrained = 0;
	}
}

// Initializers are standalone functions and not network attributes
// Therefore, the initialization functions being used need not be specified in the network

#endif
