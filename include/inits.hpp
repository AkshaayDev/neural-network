#ifndef INITS_HPP
#define INITS_HPP

#include <chrono>

namespace NNInitialisations {
	// Xavier initialisation
	// Initialise weights to random value between +- sqrt(6/(n_in + n_out))
	void xavierInitialisation(NeuralNetwork& nn) {
		std::mt19937 gen(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
		for (int i = 0; i < nn.weights.size(); i++) {
			double limit = std::sqrt(6.0 / (nn.layers[i] + nn.layers[i + 1]));
			std::uniform_real_distribution<> dis(-limit, limit);
			nn.weights[i].forEach([&dis, &gen](double *val, int, int) {
				*val = static_cast<double>(dis(gen));
			});
		}
	}
};

#endif
