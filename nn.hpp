#ifndef NN_HPP
#define NN_HPP

class NeuralNetwork {
public:
	std::vector<int> layers;
	// Represents the structure of the neural network (How many neurons is in each layer)
	int depth; // Number of layers

	std::vector<NNMatrix> weights;
	// weights[i] are the weights connecting layer i to layer i+1
	// weights.size() == depth - 1, dimension of weights[i] == layers[i+1] by layers[i]
	std::vector<NNMatrix> biases;
	// biases[i] are the biases for layer i+1
	// biases.size() == depth - 1, dimension of biases[i] == layers[i+1] by 1
	std::vector<NNMatrix> activations;
	// activations[i] are the activations of layer i
	// activations.size() == depth, dimension of activations[i] == layers[i] by 1
	std::vector<NNMatrix> rawActivations;
	// rawActivations[i] are the raw activations of layer i+1
	// rawActivations.size() == depth - 1, dimension of rawActivations[i] == layers[i+1] by 1
	std::vector<NNMatrix> DW;
	// Partial derivative of the loss with respect to the weights (same dimensions as weights)
	std::vector<NNMatrix> DB;
	// Partial derivative of the loss with respect to the biases (same dimensions as biases)

	// Functions for the network
	std::function<void(NeuralNetwork&)> initialisationFn;
	std::function<NNMatrix(NNMatrix)> hiddenActivationFn;
	std::function<NNMatrix(NNMatrix)> hiddenActivationFnDerivative;
	std::function<NNMatrix(NNMatrix)> outputActivationFn;
	std::function<NNMatrix(NNMatrix)> outputActivationFnDerivative;
	std::function<double(NNMatrix, NNMatrix)> lossFn;
	std::function<NNMatrix(NNMatrix, NNMatrix)> lossFnDerivative;

	// Set the layers of the network and resize the property matrices accordingly
	void setLayers(std::vector<int> layers) {
		this->layers = layers;
		depth = layers.size();

		weights.resize(depth - 1);
		DW.resize(depth - 1);
		biases.resize(depth - 1);
		DB.resize(depth - 1);
		activations.resize(depth);
		rawActivations.resize(depth - 1);
		for (int i = 0; i < depth; i++) {
			if (i != depth - 1) {
				weights[i].resize(layers[i + 1], layers[i]);
				DW[i].resize(layers[i + 1], layers[i]);
				biases[i].resize(layers[i + 1], 1);
				DB[i].resize(layers[i + 1], 1);
				rawActivations[i].resize(layers[i + 1], 1);
			}
			activations[i].resize(layers[i], 1);
		}
	}
	// Use the predefined initialisation function to initialise the parameters
	void initialise() { initialisationFn(*this); }
	// Sets activations and raw activations after forward propogation of the input
	void forwardPropagation(NNMatrix input) {
		activations[0] = input;
		for (int i = 0; i < depth - 1; i++) {
			// Z_i = W_i . A_i + B_i
			rawActivations[i] = NNMatrix::dot(weights[i], activations[i]) + biases[i];
			// A_i+1 = f(Z_i)
			if (i == depth - 2) {
				activations[i + 1] = outputActivationFn(rawActivations[i]);
			} else {
				activations[i + 1] = hiddenActivationFn(rawActivations[i]);
			}
		}
	}
	// Forward propogate the input but return only the output and do not set any activations
	NNMatrix run(NNMatrix input) {
		for (int i = 0; i < depth - 1; i++) {
			// A_i+1 = f(W_i . A_i + B_i)
			input = NNMatrix::dot(weights[i], input) + biases[i];
			input = (i == depth - 2) ? outputActivationFn(input) : hiddenActivationFn(input);
		}
		return input;
	}
	// Sets the partial derivatives of the loss with respect to the weights and biases
	void backwardPropagation(NNMatrix input, NNMatrix target) {
		forwardPropagation(input);
		for (int i = DB.size() - 1; i >= 0; i--) {
			if (i == DB.size() - 1) {
				NNMatrix lossDerivative = lossFnDerivative(activations.back(), target);
				// ∂L/∂A_depth-1 = L'(A_depth-1, target)
				DB[i] = lossDerivative * outputActivationFnDerivative(rawActivations[i]);
				// DB_depth-2 = ∂L/∂A_depth-1 * f'(Z_depth-2)
			} else {
				// DB_i = (W_i+1^T . DB_i+1) * f'(Z_i)
				DB[i] = NNMatrix::dot(weights[i + 1].transpose(), DB[i + 1]) * hiddenActivationFnDerivative(rawActivations[i]);
			}
			// DW_i = DB_i . A_i^T
			DW[i] = NNMatrix::dot(DB[i], activations[i].transpose());
		}
	}
	// Train the network using gradient descent
	void batchGradientDescent(double learningRate, int iterations, std::vector<std::pair<NNMatrix, NNMatrix>> batch) {
		for (int epoch = 0; epoch < iterations; epoch++) {
			std::vector<NNMatrix> totalDW, totalDB;
			// These are the total partial derivatives of the loss with respect to the weights and biases
			// Resize them to have the same dimensions as the weights and biases
			totalDW.resize(depth - 1);
			totalDB.resize(depth - 1);
			for (int i = 0; i < depth - 1; i++) {
				totalDW[i].resize(layers[i + 1], layers[i]);
				totalDB[i].resize(layers[i + 1], 1);
			}
			// Accumulate the partial derivatives for each sample in the batch
			for (std::pair<NNMatrix, NNMatrix> sample : batch) {
				backwardPropagation(sample.first, sample.second);
				for (int i = 0; i < depth - 1; i++) {
					totalDW[i] = totalDW[i] + DW[i];
					totalDB[i] = totalDB[i] + DB[i];
				}
			}
			// Using the accumulated partial derivatives, find the average and update the parameters
			// θ = θ - α * ∂L/∂θ
			for (int i = 0; i < depth - 1; i++) {
				weights[i] = weights[i] - (totalDW[i] / batch.size()) * learningRate;
				biases[i] = biases[i] - (totalDB[i] / batch.size()) * learningRate;
			}
		}
	}
};

#endif
