#ifndef TRAINER_HPP
#define TRAINER_HPP

namespace NNTrainer {
	// Train the network using gradient descent
	// Hyperparameters: { "learning_rate": double, "iterations": int }
	void gradientDescent(NeuralNetwork& nn, std::vector<std::pair<NNMatrix, NNMatrix>> batch, std::unordered_map<std::string, double> hyperparams) {
		double learningRate = hyperparams["learning_rate"];
		int iterations = hyperparams["iterations"];

		for (int epoch = 0; epoch < iterations; epoch++) {
			std::vector<NNMatrix> totalDW, totalDB;
			// These are the total partial derivatives of the loss with respect to the weights and biases
			// Resize them to have the same dimensions as the weights and biases
			totalDW.resize(nn.depth - 1);
			totalDB.resize(nn.depth - 1);
			for (int i = 0; i < nn.depth - 1; i++) {
				totalDW[i].resize(nn.layers[i + 1], nn.layers[i]);
				totalDB[i].resize(nn.layers[i + 1], 1);
			}
			// Accumulate the partial derivatives for each sample in the batch
			for (std::pair<NNMatrix, NNMatrix> sample : batch) {
				nn.backwardPropagation(sample.first, sample.second);
				for (int i = 0; i < nn.depth - 1; i++) {
					totalDW[i] = totalDW[i] + nn.DW[i];
					totalDB[i] = totalDB[i] + nn.DB[i];
				}
			}
			// Using the accumulated partial derivatives, find the average and update the parameters
			// θ = θ - α * ∂L/∂θ
			for (int i = 0; i < nn.depth - 1; i++) {
				nn.weights[i] = nn.weights[i] - (totalDW[i] / batch.size()) * learningRate;
				nn.biases[i] = nn.biases[i] - (totalDB[i] / batch.size()) * learningRate;
			}
		}
	}
	// Train the network using momentum
	// Hyperparameters: { "learning_rate": double, "beta": double, "iterations": int }
	void momentum(NeuralNetwork& nn, std::vector<std::pair<NNMatrix, NNMatrix>> batch, std::unordered_map<std::string, double> hyperparams) {
		double learningRate = hyperparams["learning_rate"];
		double beta = hyperparams["beta"];
		int iterations = hyperparams["iterations"];

		std::vector<NNMatrix> VW, VB;
		// These are the velocity terms of the partial derivatives
		// Resize them to have the same dimensions as the weights and biases
		VW.resize(nn.depth - 1);
		VB.resize(nn.depth - 1);
		for (int i = 0; i < nn.depth - 1; i++) {
			VW[i].resize(nn.layers[i + 1], nn.layers[i]);
			VB[i].resize(nn.layers[i + 1], 1);
		}

		for (int epoch = 0; epoch < iterations; epoch++) {
			std::vector<NNMatrix> totalDW, totalDB;
			// These are the total partial derivatives of the loss with respect to the weights and biases
			// Resize them to have the same dimensions as the weights and biases
			totalDW.resize(nn.depth - 1);
			totalDB.resize(nn.depth - 1);
			for (int i = 0; i < nn.depth - 1; i++) {
				totalDW[i].resize(nn.layers[i + 1], nn.layers[i]);
				totalDB[i].resize(nn.layers[i + 1], 1);
			}
			// Accumulate the partial derivatives for each sample in the batch
			for (std::pair<NNMatrix, NNMatrix> sample : batch) {
				nn.backwardPropagation(sample.first, sample.second);
				for (int i = 0; i < nn.depth - 1; i++) {
					totalDW[i] = totalDW[i] + nn.DW[i];
					totalDB[i] = totalDB[i] + nn.DB[i];
				}
			}
			// Using the accumulated partial derivatives, find the average and update the parameters and their velocity
			// v = β * v + (1 - β) * ∂L/∂θ
			// θ = θ + α * v
			for (int i = 0; i < nn.depth - 1; i++) {
				VW[i] = VW[i] * beta + (totalDW[i] / batch.size()) * (1 - beta);
				VB[i] = VB[i] * beta + (totalDB[i] / batch.size()) * (1 - beta);
				nn.weights[i] = nn.weights[i] + VW[i] * learningRate;
				nn.biases[i] = nn.biases[i] + VB[i] * learningRate;
			}
		}
	}
	// Train the network using adam (adaptive moment estimation)
	// Hyperparameters: { "learning_rate": double, "beta1": double, "beta2": double, "epsilon": double, "iterations": int }
	void adam(NeuralNetwork& nn, std::vector<std::pair<NNMatrix, NNMatrix>> batch, std::unordered_map<std::string, double> hyperparams) {
		double learningRate = hyperparams["learning_rate"];
		double beta1 = hyperparams["beta1"];
		double beta2 = hyperparams["beta2"];
		double epsilon = hyperparams["epsilon"];
		int iterations = hyperparams["iterations"];

		std::vector<NNMatrix> MW, MB, VW, VB;
		// These are the momentum and velocity terms of the partial derivatives
		// The momentum measures the partial derivative
		// The velocity measures the square of the partial derivative
		// Resize them to have the same dimensions as the weights and biases
		MW.resize(nn.depth - 1);
		MB.resize(nn.depth - 1);
		VW.resize(nn.depth - 1);
		VB.resize(nn.depth - 1);
		for (int i = 0; i < nn.depth - 1; i++) {
			MW[i].resize(nn.layers[i + 1], nn.layers[i]);
			MB[i].resize(nn.layers[i + 1], 1);
			VW[i].resize(nn.layers[i + 1], nn.layers[i]);
			VB[i].resize(nn.layers[i + 1], 1);
		}

		for (int epoch = 0; epoch < iterations; epoch++) {
			std::vector<NNMatrix> totalDW, totalDB;
			// These are the total partial derivatives of the loss with respect to the weights and biases
			// Resize them to have the same dimensions as the weights and biases
			totalDW.resize(nn.depth - 1);
			totalDB.resize(nn.depth - 1);
			for (int i = 0; i < nn.depth - 1; i++) {
				totalDW[i].resize(nn.layers[i + 1], nn.layers[i]);
				totalDB[i].resize(nn.layers[i + 1], 1);
			}
			// Accumulate the partial derivatives for each sample in the batch
			for (std::pair<NNMatrix, NNMatrix> sample : batch) {
				nn.backwardPropagation(sample.first, sample.second);
				for (int i = 0; i < nn.depth - 1; i++) {
					totalDW[i] = totalDW[i] + nn.DW[i];
					totalDB[i] = totalDB[i] + nn.DB[i];
				}
			}
			// Using the accumulated partial derivatives, find the average and update the parameters and their momentum and velocity
			// m = β1 * m + (1 - β1) * ∂L/∂θ
			// v = β2 * v + (1 - β2) * (∂L/∂θ)^2
			// m̂ = m / (1 - (β1)^t)
			// v̂ = v / (1 - (β2)^t)
			// θ = θ - α * m̂/(sqrt(v̂) + ε)
			for (int i = 0; i < nn.depth - 1; i++) {
				NNMatrix avgDW = totalDW[i] / batch.size(), avgDB = totalDB[i] / batch.size();
				MW[i] = MW[i] * beta1 + avgDW * (1 - beta1);
				MB[i] = MB[i] * beta1 + avgDB * (1 - beta1);
				VW[i] = VW[i] * beta2 + (avgDW^2) * (1 - beta2);
				VB[i] = VB[i] * beta2 + (avgDB^2) * (1 - beta2);
				// Correction coeffecients
				double c1 = 1 - std::pow(beta1, epoch + 1);
				double c2 = 1 - std::pow(beta2, epoch + 1);
				nn.weights[i] = nn.weights[i] - ((MW[i]/c1) / ((VW[i]/c2)^0.5 + epsilon)) * learningRate;
				nn.biases[i] = nn.biases[i] - ((MB[i]/c1) / ((VB[i]/c2)^0.5 + epsilon)) * learningRate;
			}
		}
	}
}

#endif
