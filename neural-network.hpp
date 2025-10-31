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
	int depth = 0; // Number of layers
	int epochsTrained = 0;

	// weights[i] are the weights connecting layer i to layer i+1
	// weights.size() == depth - 1, dimension of weights[i] == layers[i+1] by layers[i]
	std::vector<NNMatrix> weights;
	// biases[i] are the biases for layer i+1
	// biases.size() == depth - 1, dimension of biases[i] == layers[i+1] by 1
	std::vector<NNMatrix> biases;
	// activations[i] are the activations of layer i
	// activations.size() == depth, dimension of activations[i] == layers[i] by 1
	std::vector<NNMatrix> activations;
	// rawActivations[i] are the raw activations of layer i+1
	// rawActivations.size() == depth - 1, dimension of rawActivations[i] == layers[i+1] by 1
	std::vector<NNMatrix> rawActivations;
	// Partial derivative of the loss with respect to the parameters (same dimensions)
	std::vector<NNMatrix> DW, DB;
	// Averaged DW and DB used for training
	std::vector<NNMatrix> avgDW, avgDB;
	// Parameter error velocities and momentums for training with momentum and adam
	std::vector<NNMatrix> VW, VB, MW, MB;

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
		biases.resize(depth - 1);
		DW.resize(depth - 1);
		avgDW.resize(depth - 1);
		DB.resize(depth - 1);
		avgDB.resize(depth - 1);
		VW.resize(depth - 1);
		VB.resize(depth - 1);
		MW.resize(depth - 1);
		MB.resize(depth - 1);
		activations.resize(depth);
		rawActivations.resize(depth - 1);
		for (int i = 0; i < depth; i++) {
			if (i != depth - 1) {
				weights[i].resize(layers[i + 1], layers[i]); weights[i].fill(0);
				biases[i].resize(layers[i + 1], 1); biases[i].fill(0);
				DW[i].resize(layers[i + 1], layers[i]);
				avgDW[i].resize(layers[i + 1], layers[i]);
				DB[i].resize(layers[i + 1], 1);
				avgDB[i].resize(layers[i + 1], 1);
				VW[i].resize(layers[i + 1], layers[i]); VW[i].fill(0);
				VB[i].resize(layers[i + 1], 1); VB[i].fill(0);
				MW[i].resize(layers[i + 1], layers[i]); MW[i].fill(0);
				MB[i].resize(layers[i + 1], 1); MB[i].fill(0);
				rawActivations[i].resize(layers[i + 1], 1); rawActivations[i].fill(0);
			}
			activations[i].resize(layers[i], 1); activations[i].fill(0);
		}
	}
	// Set the activation functions of the network
	// Pass NNActivationType as arguments
	void setActivationFunctions(std::string hidden, std::string output) {
		if (hidden == NNActivationType::Sigmoid) {
			hiddenActivationFnName = "sigmoid";
			hiddenActivationFn = NNActivation::sigmoid;
			hiddenActivationFnDerivative = NNActivation::sigmoidDerivative;
		} else if (hidden == NNActivationType::ReLU) {
			hiddenActivationFnName = "relu";
			hiddenActivationFn = NNActivation::relu;
			hiddenActivationFnDerivative = NNActivation::reluDerivative;
		} else if (hidden == NNActivationType::Tanh) {
			hiddenActivationFnName = "tanh";
			hiddenActivationFn = NNActivation::tanh;
			hiddenActivationFnDerivative = NNActivation::tanhDerivative;		
		} else throw std::runtime_error("Unknown hidden activation function ('" + hidden + "')");

		if (output == NNActivationType::Sigmoid) {
			outputActivationFnName = "sigmoid";
			outputActivationFn = NNActivation::sigmoid;
			outputActivationFnDerivative = NNActivation::sigmoidDerivative;
		} else if (output == NNActivationType::ReLU) {
			outputActivationFnName = "relu";
			outputActivationFn = NNActivation::relu;
			outputActivationFnDerivative = NNActivation::reluDerivative;
		} else if (output == NNActivationType::Tanh) {
			outputActivationFnName = "tanh";
			outputActivationFn = NNActivation::tanh;
			outputActivationFnDerivative = NNActivation::tanhDerivative;
		} else if (output == NNActivationType::Softmax) {
			outputActivationFnName = "softmax";
			outputActivationFn = NNActivation::softmax;
		} else throw std::runtime_error("Unknown output activation function ('" + output + "')");
	}
	// Set the loss function of the network
	// Pass NNLossType as argument
	void setLossFunction(std::string loss) {
		if (loss == NNLossType::MSE) {
			lossFnName = "mse";
			lossFn = NNLoss::MSE;
			lossFnDerivative = NNLoss::MSEDerivative;
		} else if (loss == NNLossType::CCE) {
			lossFnName = "cce";
			lossFn = NNLoss::CCE;
			lossFnDerivative = NNLoss::CCEDerivative;
		} else throw std::runtime_error("Unknown loss function ('" + loss + "')");
	}
	// Accumulate and average the partial derivatives for each sample in the batch
	void averagePDs(std::vector<std::pair<NNMatrix, NNMatrix>> batch) {
		for (int i = 0; i < depth - 1; i++) {
			avgDW[i].fill(0);
			avgDB[i].fill(0);
		}
		for (std::pair<NNMatrix, NNMatrix> sample : batch) {
			backwardPropagation(sample.first, sample.second);
			for (int i = 0; i < depth - 1; i++) {
				avgDW[i] = avgDW[i] + DW[i];
				avgDB[i] = avgDB[i] + DB[i];
			}
		}
		for (int i = 0; i < depth - 1; i++) {
			avgDW[i] = avgDW[i] / batch.size();
			avgDB[i] = avgDB[i] / batch.size();
		}
	}

	// Sets activations and raw activations after forward propagation of the input
	void forwardPropagation(NNMatrix input) {
		if (!NNMatrix::sameSize(input, activations[0])) throw std::runtime_error("Input matrix dimension mismatch: " +
			std::to_string(input.rows()) + "x" + std::to_string(input.cols()) + " instead of " +
			std::to_string(activations[0].rows()) + "x" + std::to_string(activations[0].cols())
		);
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
	// Forward propagate the input but return only the output and do not set any activations
	NNMatrix run(NNMatrix input) {
		if (!NNMatrix::sameSize(input, activations[0])) throw std::runtime_error("Input matrix dimension mismatch: " +
			std::to_string(input.rows()) + "x" + std::to_string(input.cols()) + " instead of " +
			std::to_string(activations[0].rows()) + "x" + std::to_string(activations[0].cols())
		);
		for (int i = 0; i < depth - 1; i++) {
			// A_i+1 = f(W_i . A_i + B_i)
			input = NNMatrix::dot(weights[i], input) + biases[i];
			input = (i == depth - 2) ? outputActivationFn(input) : hiddenActivationFn(input);
		}
		return input;
	}
	// Sets the partial derivatives of the loss with respect to the weights and biases
	void backwardPropagation(NNMatrix input, NNMatrix target) {
		if (!NNMatrix::sameSize(target, activations.back())) throw std::runtime_error("Target matrix dimension mismatch: " +
			std::to_string(target.rows()) + "x" + std::to_string(target.cols()) + " instead of " +
			std::to_string(activations.back().rows()) + "x" + std::to_string(activations.back().cols())
		);
		forwardPropagation(input);
		for (int i = DB.size() - 1; i >= 0; i--) {
			if (i == DB.size() - 1) {
				// Handle Softmax + Cross Entropy
				if (outputActivationFnName == "softmax") {
					if (lossFnName == "cce") {
						DB[i] = activations.back() - target; // Ŷ - Y
					} else throw std::runtime_error("Softmax can only be used with cross entropy loss");
				} else {
					// ∂L/∂A_depth-1 = L'(A_depth-1, target)
					NNMatrix lossDerivative = lossFnDerivative(activations.back(), target);
					// DB_depth-2 = ∂L/∂A_depth-1 * f'(Z_depth-2)
					DB[i] = lossDerivative * outputActivationFnDerivative(rawActivations[i]);
				}
			} else {
				// DB_i = (W_i+1^T . DB_i+1) * f'(Z_i)
				DB[i] = NNMatrix::dot(weights[i + 1].transpose(), DB[i + 1]) * hiddenActivationFnDerivative(rawActivations[i]);
			}
			// DW_i = DB_i . A_i^T
			DW[i] = NNMatrix::dot(DB[i], activations[i].transpose());
		}
	}
	
	// Save the parameters and architecture to an output file stream with an option to include the training state
	void save(std::ofstream& out, bool includeTrainingData = false) {
		// Write the depth
		out.write(reinterpret_cast<const char*>(&depth), sizeof(int));
		// Write the layers
		out.write(reinterpret_cast<const char*>(layers.data()), depth * sizeof(int));
		// Write the activation function and loss function names
		std::string fnNames[3] = {hiddenActivationFnName, outputActivationFnName, lossFnName};
		for (std::string& fnName : fnNames) {
			uint32_t size = fnName.size();
			out.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
			out.write(fnName.c_str(), size);
		}
		// Write the parameters
		saveMatrixVector(weights, out);
		saveMatrixVector(biases, out);
		// Write the epochs trained
		out.write(reinterpret_cast<const char*>(&epochsTrained), sizeof(int));
		// Write whether training state is included
		out.write(reinterpret_cast<const char*>(&includeTrainingData), sizeof(bool));
		// Write optional training state
		if (includeTrainingData) {
			saveMatrixVector(VW, out);
			saveMatrixVector(VB, out);
			saveMatrixVector(MW, out);
			saveMatrixVector(MB, out);
		}
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
			uint32_t size = 0;
			in.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
			fn->resize(size);
			in.read(&(*fn)[0], size);
		}
		setActivationFunctions(hiddenActivationFnName, outputActivationFnName);
		setLossFunction(lossFnName);
		// Read the parameters
		loadMatrixVector(weights, in);
		loadMatrixVector(biases, in);
		// Read the epochs trained
		in.read(reinterpret_cast<char*>(&epochsTrained), sizeof(int));
		// Read whether training state is included
		bool hasTrainingData = false;
		in.read(reinterpret_cast<char*>(&hasTrainingData), sizeof(bool));
		// Read optional training state
		if (hasTrainingData) {
			loadMatrixVector(VW, in);
			loadMatrixVector(VB, in);
			loadMatrixVector(MW, in);
			loadMatrixVector(MB, in);
		}
	}
private:
	// Helper to write a matrix vector to an output file stream (Assumes matrix dimensions are known)
	void saveMatrixVector(std::vector<NNMatrix>& vec, std::ofstream& out) {
		for (NNMatrix& mat : vec) {
			mat.forEach([&out](double *val, int, int) {
				out.write(reinterpret_cast<const char*>(val), sizeof(double));
			});
		}
	}
	// Helper to load a matrix vector from an output file stream (Assumes matrix has correct dimensions)
	void loadMatrixVector(std::vector<NNMatrix>& vec, std::ifstream& in) {
		for (NNMatrix& mat : vec) {
			for (int i = 0; i < mat.rows(); i++) {
				in.read(reinterpret_cast<char*>(mat[i].data()), mat.cols() * sizeof(double));
			}
		}
	}
};

#include "./inits.hpp"
#include "./trainer.hpp"

#endif
