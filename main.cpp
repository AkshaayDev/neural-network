#include "neural_network.hpp"

int main() {
	// We will be using a simple 2,2,1 architecture to test the XOR gate
	NeuralNetwork nn;
	nn.setLayers({2,2,1});
	nn.initialisationFn = NNInitialisations::xavierInitialisation;
	nn.initialise();
	nn.activationFn = NNActivation::sigmoid;
	nn.activationFnDerivative = NNActivation::sigmoidDerivative;
	nn.lossFn = NNLoss::MSE;
	nn.lossFnDerivative = NNLoss::MSEDerivative;
	std::vector<std::pair<NNMatrix, NNMatrix>> data = {
		{ NNMatrix(std::vector<std::vector<double>>{{0},{0}}), NNMatrix(std::vector<std::vector<double>>{{0}}) },
		{ NNMatrix(std::vector<std::vector<double>>{{0},{1}}), NNMatrix(std::vector<std::vector<double>>{{1}}) },
		{ NNMatrix(std::vector<std::vector<double>>{{1},{0}}), NNMatrix(std::vector<std::vector<double>>{{1}}) },
		{ NNMatrix(std::vector<std::vector<double>>{{1},{1}}), NNMatrix(std::vector<std::vector<double>>{{0}}) }
	};
	nn.batchGradientDescent(0.1, 100000, data);
	for (std::pair<NNMatrix, NNMatrix> sample : data) {
		std::cout << sample.first.data[0][0];
		std::cout << " ^ ";
		std::cout << sample.first.data[1][0];
		std::cout << " = ";
		NNMatrix::print(nn.run(sample.first));
	}
}
