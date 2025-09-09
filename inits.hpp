#ifndef INITS_HPP
#define INITS_HPP

#include <chrono>

namespace NNInitialisations {
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
	}
}

#endif
