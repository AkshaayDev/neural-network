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
#include <algorithm>
#include <memory>
#include "./matrix.hpp"
#include "./activation.hpp"
#include "./loss.hpp"
#include "./layer.hpp"

class NeuralNetwork {
public:
	static const uint32_t SIGNATURE = 0x4E4E4454; // "NNDT", Used for validating data files
	std::vector<std::unique_ptr<Layer>> layers;
	// Represents the structure of the neural network (How many neurons is in each layer)
	int depth = 0; // Number of layers
	int iterationsTrained = 0, epochsTrained = 0;

	// Averaged gradients of each layer
	std::vector<std::vector<Matrix>> avgGrads;
	// Momentum buffers for training
	std::vector<std::vector<Matrix>> momentumV, adamM, adamV;

	// Loss function for the network
	std::string lossFnName;
	std::function<double(Matrix, Matrix)> lossFn;
	std::function<Matrix(Matrix, Matrix)> lossFnDerivative;

	// Setters

	// Add a layer to the network (Usage: nn.addLayer<LayerType>(args); )
	template<typename LayerType, typename... Args>
	void addLayer(Args&&... args) {
		layers.push_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
		std::vector<Matrix>& lastGrads = layers.back()->grads;
		// Pushing back the gradients directly works because they have just been initialized
		avgGrads.push_back(lastGrads);
		momentumV.push_back(lastGrads);
		adamM.push_back(lastGrads);
		adamV.push_back(lastGrads);
		depth++;
	}

	// Set the loss function of the network
	// Pass LossType as argument
	void setLossFunction(std::string loss) {
		if (loss == LossType::MSE) {
			lossFn = Loss::MSE;
			lossFnDerivative = Loss::MSEDerivative;
		} else if (loss == LossType::CCE) {
			lossFn = Loss::CCE;
			lossFnDerivative = Loss::CCEDerivative;
		} else throw std::runtime_error("Unknown loss function ('" + loss + "')");
		lossFnName = loss;
	}

	// Accumulate and average the partial derivatives for each sample in the batch
	void averagePDs(std::vector<std::pair<Matrix, Matrix>> batch) {
		for (int i = 0; i < depth; i++) {
			for (Matrix& avgGrad : avgGrads[i]) {
				avgGrad.fill(0);
			}
		}
		for (std::pair<Matrix, Matrix> sample : batch) {
			Matrix predicted = forwardPropagation(sample.first);
			backwardPropagation(predicted, sample.second);
			for (int i = 0; i < depth; i++) {
				for (int j = 0; j < layers[i]->grads.size(); j++) {
					avgGrads[i][j] = avgGrads[i][j] + layers[i]->grads[j];
				}
			}
		}
		for (int i = 0; i < depth; i++) {
			for (Matrix& avgGrad : avgGrads[i]) {
				avgGrad = avgGrad / batch.size();
			}
		}
	}

	// Performs a feed forward without storing inputs or outputs
	Matrix run(Matrix input) {
		if (layers.empty()) throw std::runtime_error("Cannot run an empty network");
		for (auto& layer : layers) {
			input = layer->run(input);
		}
		return input;
	}
	// Sets layer inputs and outputs after forward propagation of an input and returns network output
	Matrix forwardPropagation(Matrix input) {
		if (layers.empty()) throw std::runtime_error("Cannot forward propagate through an empty network");
		for (auto& layer : layers) {
			input = layer->forward(input);
		}
		return input;
	}
	// Sets the layer gradients (partial derivatives of the loss with respect to its parameters)
	// Note: forward propagation has to be called first and its recommended to pass its return value as `predicted`
	void backwardPropagation(Matrix predicted, Matrix real) {
		if (layers.empty()) throw std::runtime_error("Cannot backward propagate through an empty network");
		Matrix dy = lossFnDerivative(predicted, real);
		for (int i = depth - 1; i >= 0; i--) {
			dy = layers[i]->backward(dy);
		}
	}

	// Save the parameters and architecture to an output file stream with an option to include the training state
	void save(std::ofstream& out, bool includeTrainingData = false) {
		// Write the signature
		uint32_t sig = SIGNATURE;
		out.write(reinterpret_cast<const char*>(&sig), sizeof(uint32_t));
		// Write the depth
		out.write(reinterpret_cast<const char*>(&depth), sizeof(int));
		// Write the layers
		for (int i = 0; i < depth; i++) {
			layers[i]->save(out);
		}
		// Write the loss function
		uint32_t size = lossFnName.size();
		out.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
		out.write(lossFnName.c_str(), size);
		// Write the iterations and epochs trained
		out.write(reinterpret_cast<const char*>(&iterationsTrained), sizeof(int));
		out.write(reinterpret_cast<const char*>(&epochsTrained), sizeof(int));
		// Write whether training state is included
		out.write(reinterpret_cast<const char*>(&includeTrainingData), sizeof(bool));
		if (includeTrainingData) {
			saveTrainingMoment(momentumV, out);
			saveTrainingMoment(adamM, out);
			saveTrainingMoment(adamV, out);
		}
	}
	
	// Load the parameters and architecture from an input file stream
	void load(std::ifstream& in) {
		// Read the signature
		uint32_t sig;
		in.read(reinterpret_cast<char*>(&sig), sizeof(uint32_t));
		if (sig != SIGNATURE) throw std::runtime_error("Invalid signature read.");
		// Clear all vector attributes
		layers.clear();
		avgGrads.clear();
		momentumV.clear();
		adamM.clear();
		adamV.clear();
		// Read the depth
		in.read(reinterpret_cast<char*>(&depth), sizeof(int));
		// Read the layers
		for (int i = 0; i < depth; i++) {
			std::unique_ptr<Layer> layer = Layer::load(in);
			layers.emplace_back(std::move(layer));
			std::vector<Matrix>& lastGrads = layers.back()->grads;
			// Pushing back the gradients directly works because they have just been initialized
			avgGrads.push_back(lastGrads);
			momentumV.push_back(lastGrads);
			adamM.push_back(lastGrads);
			adamV.push_back(lastGrads);
		}
		// Read the loss function
		uint32_t size = 0;
		in.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
		lossFnName.resize(size);
		in.read(&lossFnName[0], size);
		setLossFunction(lossFnName);
		// Read the iterations and epochs trained
		in.read(reinterpret_cast<char*>(&iterationsTrained), sizeof(int));
		in.read(reinterpret_cast<char*>(&epochsTrained), sizeof(int));
		// Read whether training data in included
		bool hasTrainingData = false;
		in.read(reinterpret_cast<char*>(&hasTrainingData), sizeof(bool));
		if (hasTrainingData) {
			loadTrainingMoment(momentumV, in);
			loadTrainingMoment(adamM, in);
			loadTrainingMoment(adamV, in);
		}
	}
private:
	// Helper to write a moment tensor to an output file stream (Assumes tensor dimensions are known)
	void saveTrainingMoment(std::vector<std::vector<Matrix>>& moment, std::ofstream& out) {
		for (std::vector<Matrix>& layerMoment : moment) {
			for (Matrix& gradMoment : layerMoment) {
				gradMoment.forEach([&out](double *val, int, int) {
					out.write(reinterpret_cast<const char*>(val), sizeof(double));
				});
			}
		}
	}
	// Helper to read a moment tensor from an input file stream (Assumes tensor has correct dimensions)
	void loadTrainingMoment(std::vector<std::vector<Matrix>>& moment, std::ifstream& in) {
		for (std::vector<Matrix>& layerMoment : moment) {
			for (Matrix& gradMoment : layerMoment) {
				for (int i = 0; i < gradMoment.rows(); i++) {
					in.read(reinterpret_cast<char*>(gradMoment[i].data()), gradMoment.cols() * sizeof(double));
				}
			}
		}
	}
};

#include "./inits.hpp"
#include "./trainer.hpp"

#endif
