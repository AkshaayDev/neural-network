#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "./neural-network.hpp"

enum class OptimizerType { GradientDescent, Momentum, Adam };

class Trainer {
public:
	NeuralNetwork& nn;
	std::vector<std::pair<Matrix, Matrix>>& batch;
	std::function<void()> iterationCallback = []() {};
	std::function<void()> epochCallback = []() {};
	// Network reference and batch reference constructor
	Trainer(NeuralNetwork& nn, std::vector<std::pair<Matrix, Matrix>>& batch) : nn(nn), batch(batch) {}
	// Used for gradient descent, momentum and adam (Default 0.001)
	double learningRate = 0.001;
	// Used for momentum (Default 0.9)
	double beta = 0.9;
	// Used for adam (Default 0.9)
	double beta1 = 0.9;
	// Used for adam (Default 0.999)
	double beta2 = 0.999;
	// Used for adam (Default 1e-8)
	double epsilon = 1e-8;
	// Training data is split into smaller samples of `sampleSize` to be processed individually
	// sampleSize is -1 by default, meaning the whole batch is processed at once
	int sampleSize = -1;
	// Training data is shuffled before every epoch by default
	bool enableShuffling = true;

	// Train the network
	void train(OptimizerType optimizer, int epochs, unsigned int shuffleSeed = std::chrono::system_clock::now().time_since_epoch().count()) {
		std::mt19937 gen(shuffleSeed);
		int actualSize = (sampleSize == -1) ? batch.size() : sampleSize;

		for (int epoch = 1; epoch <= epochs; epoch++) {
			if (enableShuffling) std::shuffle(batch.begin(), batch.end(), gen);
			for (int i = 0; i < batch.size(); i += actualSize) {
				std::vector<std::pair<Matrix, Matrix>> sample(
					batch.begin() + i,
					batch.begin() + std::min(i + actualSize, static_cast<int>(batch.size()))
				);
				nn.averagePDs(sample);
				switch (optimizer) {
					case OptimizerType::GradientDescent: gradientDescent(); break;
					case OptimizerType::Momentum: momentum(); break;
					case OptimizerType::Adam: adam(); break;
					default: throw std::runtime_error("Unknown optimizer value");
				}
				nn.iterationsTrained++;
				iterationCallback();
			}
			nn.epochsTrained++;
			epochCallback();
		}
	}
	// Note: all these functions assume that the average partial derivatives are already set and iteration, epoch and callback are handled in train function
	// Train the network using gradient descent (Requires learningRate)
	inline void gradientDescent() {
		// θ = θ - α * ∂L/∂θ
		for (int i = 0; i < nn.depth; i++) {
			for (int j = 0; j < nn.layers[i]->params.size(); j++) {
				Matrix& param = nn.layers[i]->params[j];
				Matrix& avgGrad = nn.avgGrads[i][j];
				param = param - learningRate * avgGrad;
			}
		}
	}
	// Train the network using momentum (Requires learningRate, beta)
	inline void momentum() {
		// v = β * v + (1 - β) * ∂L/∂θ
		// θ = θ - α * v
		for (int i = 0; i < nn.depth; i++) {
			for (int j = 0; j < nn.layers[i]->params.size(); j++) {
				Matrix& param = nn.layers[i]->params[j];
				Matrix& avgGrad = nn.avgGrads[i][j];
				Matrix& v = nn.momentumV[i][j];
				v = beta * v + (1 - beta) * avgGrad;
				param = param - learningRate * v;
			}
		}
	}
	// Train the network using adam (adaptive moment estimation) (Requires learningRate, beta1, beta2, epsilon)
	inline void adam() {
		// m = β1 * m + (1 - β1) * ∂L/∂θ
		// v = β2 * v + (1 - β2) * (∂L/∂θ)^2
		// m̂ = m / (1 - (β1)^t)
		// v̂ = v / (1 - (β2)^t)
		// θ = θ - α * m̂/(sqrt(v̂) + ε)
		// Correction coeffecients
		double c1 = 1 - std::pow(beta1, nn.iterationsTrained + 1);
		double c2 = 1 - std::pow(beta2, nn.iterationsTrained + 1);
		for (int i = 0; i < nn.depth; i++) {
			for (int j = 0; j < nn.layers[i]->params.size(); j++) {
				Matrix& param = nn.layers[i]->params[j];
				Matrix& avgGrad = nn.avgGrads[i][j];
				Matrix& m = nn.adamM[i][j];
				Matrix& v = nn.adamV[i][j];
				m = beta1 * m + (1 - beta1) * avgGrad;
				v = beta2 * v + (1 - beta2) * (avgGrad ^ 2);
				param = param - learningRate * (m/c1) / (((v/c2) ^ 0.5) + epsilon);
			}
		}
	}
};

// Trainers are standalone functions and not network attributes
// Therefore, the trainer function being used needs not be specified in the network

#endif
