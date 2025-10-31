// You are recommended to view `examples/xor/main.cpp` first
#include "../../neural-network.hpp"
// Include data from the image hpp file
#include "img/img.hpp"
// Include fstream library for saving and loading the network data
#include <fstream>
// Include iomanip library to format .txt output
#include <iomanip>

NeuralNetwork nn;

// Helper function to create the neural network representation of the image
void createImage(std::string name) {
	std::ofstream outFile(name + ".txt");
	outFile << std::fixed << std::setprecision(10);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			double y = static_cast<double>(i) / (height - 1) * 2 - 1;
			double x = static_cast<double>(j) / (width - 1) * 2 - 1;
			outFile << nn.run(NNMatrix::fromVector({y, x})).data[0][0];
			// Add a comma between each value except the last one
			if (j != width - 1) outFile << ",";
		}
		// Add a newline between each row except the last one
		if (i != height - 1) outFile << "\n";
	}
	outFile.close();
	// Convert the .txt file to a .jpg file using display.js
	(void)std::system(("node display.js " + name).c_str());
}

void callback() {
	int epoch = nn.iterationsTrained;
	std::cout << "Iteration " << epoch << "\n";
}

std::vector<std::pair<NNMatrix, NNMatrix>> batch;

void loadBatch() {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			std::pair<NNMatrix, NNMatrix> sample;
			double y = static_cast<double>(i) / (height - 1) * 2 - 1;
			double x = static_cast<double>(j) / (width - 1) * 2 - 1;
			sample.first = NNMatrix::fromVector({y, x});
			sample.second = NNMatrix::fromScalar(expected[i * width + j]);
			batch.push_back(sample);
		}
	}
}

// In this example, we will try to get the network to learn an image with an Implicit Neural Representation(INR)
// To do this, the image is first squished into a square with sides normalised to (-1, 1)
// Then, it approximates a function f(x,y) => [pixel grayscale value at (x,y)]
// First, the image is converted to a .hpp file with a flattened array of grayscale values using `create.js`
// Then, the data is loaded into this script which creates a network or loads a pre-existing one from `nn.dat`
// After training, the network is saved to `nn.dat` and the output image is created with `display.js`
int main() {
	nn.setLayers({2,32,32,1});
	NNInitialization::xavierNormal(nn);
	nn.setActivationFunctions(NNActivationType::Tanh, NNActivationType::Sigmoid);
	nn.setLossFunction(NNLossType::MSE);
	// If there exists a data file `nn.dat`, read from it
	std::ifstream in("nn.dat", std::ios::binary);
	if (in.good()) nn.load(in);
	in.close();
	// Train the network with adam
	loadBatch();
	NNTrainer::adam(nn, batch, {
		{ "learning_rate", 0.1 },
		{ "beta1", 0.9 },
		{ "beta2", 0.999 },
		{ "epsilon", 1e-8 },
		{ "iterations", 100 }
	}, callback);
	std::cout << "Training finished." << std::endl;
	// Create the output image file `res.jpg`
	createImage("res");
	// Write network data to `nn.dat`
	std::ofstream out("nn.dat", std::ios::binary);
	nn.save(out, true);
	out.close();
}
