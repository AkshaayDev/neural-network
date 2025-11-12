// Include neural network library header from parent directory
#include "../../neural-network.hpp"
// Include fstream library for saving the network data after training
#include <fstream>

// In this example, we will be using a simple 2,2,2,1 architecture to approximate an XOR gate
int main() {
	NeuralNetwork nn;
	nn.setLayers({2,2,2,1});
	// Initialize weights using the Xavier Normal initialization
	NNInitialization::xavierNormal(nn);
	// Use sigmoid activation function for hidden and output layers
	nn.setActivationFunctions(NNActivationType::Sigmoid, NNActivationType::Sigmoid);
	// Use Mean Squared Error loss function
	nn.setLossFunction(NNLossType::MSE);
	// Set training data for the network
	std::vector<std::pair<NNMatrix, NNMatrix>> data = {
		{ NNMatrix::fromVector({0,0}), NNMatrix::fromScalar(0.0) }, // 0 ^ 0 = 0
		{ NNMatrix::fromVector({0,1}), NNMatrix::fromScalar(1.0) }, // 0 ^ 1 = 1
		{ NNMatrix::fromVector({1,0}), NNMatrix::fromScalar(1.0) }, // 1 ^ 0 = 1
		{ NNMatrix::fromVector({1,1}), NNMatrix::fromScalar(0.0) }  // 1 ^ 1 = 0
	};
	// Train the network with the training data with gradient descent
	NNTrainer trainer(nn, data);
	trainer.learningRate = 10;
	trainer.enableShuffling = false;
	trainer.train(NNOptimizerType::GradientDescent, 1000);
	// Test the network by running each test data
	for (std::pair<NNMatrix, NNMatrix> sample : data) {
		std::cout << sample.first[0][0];
		std::cout << " ^ ";
		std::cout << sample.first[1][0];
		std::cout << " = ";
		std::cout << nn.run(sample.first)[0][0] << '\n';
	}
	// Write the neural network data to a file `./out.dat` with binary encoding
	std::ofstream out("./out.dat", std::ios::binary);
	nn.save(out);
	out.close();
}

/*
On average, the output should look something like:
0 ^ 0 = 0.0187684
0 ^ 1 = 0.981912
1 ^ 0 = 0.977819
1 ^ 1 = 0.0178517
However, sometimes the training is not good enough when there are bad initializations
When this happens, the network struggles to learn the data and may output something like:
0 ^ 0 = 0.334147
0 ^ 1 = 0.332994
1 ^ 0 = 0.974527
1 ^ 1 = 0.333746
*/
