#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "./neural-network.hpp"

namespace NNTrainer {
	// Train the network using gradient descent
	// Hyperparameters: { "learning_rate": double, "iterations": int }
	inline void gradientDescent(
		NeuralNetwork& nn,
		std::vector<std::pair<NNMatrix, NNMatrix>> batch,
		std::unordered_map<std::string, double> hyperparams,
		const std::function<void()>& callback
	) {
		double learningRate = hyperparams.at("learning_rate");
		int iterations = hyperparams.at("iterations");

		for (int iteration = 0; iteration < iterations; iteration++) {
			// θ = θ - α * ∂L/∂θ
			nn.averagePDs(batch);
			for (int i = 0; i < nn.depth - 1; i++) {
				nn.weights[i] = nn.weights[i] - nn.avgDW[i] * learningRate;
				nn.biases[i] = nn.biases[i] - nn.avgDB[i] * learningRate;
			}
			nn.iterationsTrained++;
			callback();
		}
	}
	// Train the network using momentum
	// Hyperparameters: { "learning_rate": double, "beta": double, "iterations": int }
	inline void momentum(
			NeuralNetwork& nn,
			std::vector<std::pair<NNMatrix, NNMatrix>> batch,
			std::unordered_map<std::string, double> hyperparams,
			const std::function<void()>& callback
		) {
		double learningRate = hyperparams.at("learning_rate");
		double beta = hyperparams.at("beta");
		int iterations = hyperparams.at("iterations");

		for (int iteration = 0; iteration < iterations; iteration++) {
			nn.averagePDs(batch);
			// v = β * v + (1 - β) * ∂L/∂θ
			// θ = θ - α * v
			for (int i = 0; i < nn.depth - 1; i++) {
				nn.VW[i] = nn.VW[i] * beta + nn.avgDW[i] * (1 - beta);
				nn.VB[i] = nn.VB[i] * beta + nn.avgDB[i] * (1 - beta);
				nn.weights[i] = nn.weights[i] - nn.VW[i] * learningRate;
				nn.biases[i] = nn.biases[i] - nn.VB[i] * learningRate;
			}
			nn.iterationsTrained++;
			callback();
		}
	}
	// Train the network using adam (adaptive moment estimation)
	// Hyperparameters: { "learning_rate": double, "beta1": double, "beta2": double, "epsilon": double, "iterations": int }
	inline void adam(
			NeuralNetwork& nn,
			std::vector<std::pair<NNMatrix, NNMatrix>> batch,
			std::unordered_map<std::string, double> hyperparams,
			const std::function<void()>& callback
		) {
		double learningRate = hyperparams.at("learning_rate");
		double beta1 = hyperparams.at("beta1");
		double beta2 = hyperparams.at("beta2");
		double epsilon = hyperparams.at("epsilon");
		int iterations = hyperparams.at("iterations");

		for (int iteration = 0; iteration < iterations; iteration++) {
			nn.averagePDs(batch);
			// m = β1 * m + (1 - β1) * ∂L/∂θ
			// v = β2 * v + (1 - β2) * (∂L/∂θ)^2
			// m̂ = m / (1 - (β1)^t)
			// v̂ = v / (1 - (β2)^t)
			// θ = θ - α * m̂/(sqrt(v̂) + ε)
			// Correction coeffecients
			double c1 = 1 - std::pow(beta1, nn.iterationsTrained + 1);
			double c2 = 1 - std::pow(beta2, nn.iterationsTrained + 1);
			for (int i = 0; i < nn.depth - 1; i++) {
				nn.MW[i] = nn.MW[i] * beta1 + nn.avgDW[i] * (1 - beta1);
				nn.MB[i] = nn.MB[i] * beta1 + nn.avgDB[i] * (1 - beta1);
				nn.VW[i] = nn.VW[i] * beta2 + (nn.avgDW[i]^2) * (1 - beta2);
				nn.VB[i] = nn.VB[i] * beta2 + (nn.avgDB[i]^2) * (1 - beta2);
				nn.weights[i] = nn.weights[i] - ((nn.MW[i]/c1) / (((nn.VW[i]/c2)^0.5) + epsilon)) * learningRate;
				nn.biases[i] = nn.biases[i] - ((nn.MB[i]/c1) / (((nn.VB[i]/c2)^0.5) + epsilon)) * learningRate;
			}
			nn.iterationsTrained++;
			callback();
		}
	}
}

// Trainers are standalone functions and not network properties
// Therefore, the trainer function being used needs not be specified in the network

#endif
