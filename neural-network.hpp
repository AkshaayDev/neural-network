#ifndef NN_HPP
#define NN_HPP

#include <chrono>
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <random>
#include <unordered_map>
#include <string>
#include <fstream>
#include "./matrix.hpp"
#include "./activation.hpp"
#include "./loss.hpp"

class NeuralNetwork {
public:
	std::vector<int> layers;
	// Represents the structure of the neural network (How many neurons is in each layer)
	int depth; // Number of layers
	int epochsTrained = 0;

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

	std::string hiddenActivationFnName;
	std::function<NNMatrix(NNMatrix)> hiddenActivationFn;
	std::function<NNMatrix(NNMatrix)> hiddenActivationFnDerivative;
	std::string outputActivationFnName;
	std::function<NNMatrix(NNMatrix)> outputActivationFn;
	std::function<NNMatrix(NNMatrix)> outputActivationFnDerivative;
	std::string lossFnName;
	std::function<double(NNMatrix, NNMatrix)> lossFn;
	std::function<NNMatrix(NNMatrix, NNMatrix)> lossFnDerivative;

	// Setters

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
	// Set the activation functions of the network
	// Pass NNActivationType as arguments
	void setActivationFunctions(std::string hidden, std::string output) {
		hiddenActivationFnName = hidden;
		if (hidden == NNActivationType::Sigmoid) {
			hiddenActivationFn = NNActivation::sigmoid;
			hiddenActivationFnDerivative = NNActivation::sigmoidDerivative;
		} else if (hidden == NNActivationType::ReLU) {
			hiddenActivationFn = NNActivation::relu;
			hiddenActivationFnDerivative = NNActivation::reluDerivative;
		} else if (hidden == NNActivationType::Tanh) {
			hiddenActivationFn = NNActivation::tanh;
			hiddenActivationFnDerivative = NNActivation::tanhDerivative;
		}
		outputActivationFnName = output;
		if (output == NNActivationType::Sigmoid) {
			outputActivationFn = NNActivation::sigmoid;
			outputActivationFnDerivative = NNActivation::sigmoidDerivative;
		} else if (output == NNActivationType::ReLU) {
			outputActivationFn = NNActivation::relu;
			outputActivationFnDerivative = NNActivation::reluDerivative;
		} else if (output == NNActivationType::Tanh) {
			outputActivationFn = NNActivation::tanh;
			outputActivationFnDerivative = NNActivation::tanhDerivative;
		}
	}
	// Set the loss function of the network
	// Pass NNLossType as argument
	void setLossFunction(std::string loss) {
		if (loss == NNLossType::MSE) {
			lossFnName = "mse";
			lossFn = NNLoss::MSE;
			lossFnDerivative = NNLoss::MSEDerivative;
		}
	}

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

	// Save the parameters and architecture to an output file stream
	void save(std::ofstream& out) {
		// Write the depth
		out.write(reinterpret_cast<const char*>(&depth), sizeof(int));
		// Write the layers
		out.write(reinterpret_cast<const char*>(layers.data()), depth * sizeof(int));
		// Write the activation function and loss function names
		std::string fnNames[3] = {hiddenActivationFnName, outputActivationFnName, lossFnName};
		for (std::string& fnName : fnNames) {
			size_t size = fnName.size();
			out.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
			out.write(fnName.c_str(), size);
		}
		// Write the weights
		for (NNMatrix& mat : weights) {
			mat.forEach([&out](double *val, int, int) {
				out.write(reinterpret_cast<const char*>(val), sizeof(double));
			});
		}
		// Write the biases as a vector of flattened column matrices
		for (NNMatrix& mat : biases) {
			for (int i = 0; i < mat.rows(); i++) {
				out.write(reinterpret_cast<const char*>(&mat[i][0]), sizeof(double));
			}
		}
		// Write the epochs trained
		out.write(reinterpret_cast<const char*>(&epochsTrained), sizeof(int));
	}
	// Load the parameters and architecture from an input file stream
	void load(std::ifstream& in) {
		// Read the depth
		in.read(reinterpret_cast<char*>(&depth), sizeof(int));
		// Read the layers
		layers.resize(depth);
		in.read(reinterpret_cast<char*>(layers.data()), depth * sizeof(int));
		setLayers(layers);
		// Read the activation function and loss function names
		std::string* fnNames[3] = {&hiddenActivationFnName, &outputActivationFnName, &lossFnName};
		for (std::string* fn : fnNames) {
			size_t size = 0;
			in.read(reinterpret_cast<char*>(&size), sizeof(size_t));
			fn->resize(size);
			in.read(&(*fn)[0], size);
		}
		setActivationFunctions(hiddenActivationFnName, outputActivationFnName);
		setLossFunction(lossFnName);
		// Read the weights
		for (int i = 0; i < depth - 1; i++) {
			for (int j = 0; j < layers[i + 1]; j++) {
				in.read(reinterpret_cast<char*>(weights[i][j].data()), layers[i] * sizeof(double));
			}
		}
		// Read the biases as a vector of flattened column matrices
		for (int i = 0; i < depth - 1; i++) {
			std::vector<double> flattenedBiases(layers[i + 1]);
			in.read(reinterpret_cast<char*>(flattenedBiases.data()), layers[i + 1] * sizeof(double));
			biases[i] = NNMatrix::fromVector(flattenedBiases);
		}
		// Read the epochs trained
		in.read(reinterpret_cast<char*>(&epochsTrained), sizeof(int));
	}
};

#include "./inits.hpp"
#include "./trainer.hpp"

#endif