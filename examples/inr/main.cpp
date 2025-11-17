// You are recommended to view `examples/xor/main.cpp` first
#include "../../neural-network.hpp"
// Include fstream library for saving and loading the network data
#include <fstream>
// Include stb_image and stb_image_write.h to load and write image data
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
// Include OpenMP for parallel pixel processing
#include <omp.h>

NeuralNetwork nn;
const char* imgPath = "./img/img.png";
const char* outPath = "./res.png";
int width, height;
const int outWidth = 512, outHeight = 512;

// Create the neural representation of the image and save it into `outPath`
void createImage() {
	unsigned char* data = new unsigned char[outWidth * outHeight * 3];
	#pragma omp parallel for collapse(2) // Parallelize each iteration
	for (int i = 0; i < outHeight; i++) {
		for (int j = 0; j < outWidth; j++) {
			int idx = i * outWidth + j;
			double y = static_cast<double>(i) / (outHeight - 1) * 2 - 1;
			double x = static_cast<double>(j) / (outWidth - 1) * 2 - 1;
			NNMatrix rgb = nn.run(NNMatrix::fromVector({y, x}));
			for (int c = 0; c < 3; c++) {
				data[3 * idx + c] = rgb[c][0] * 255;
			}
		}
	}
	stbi_write_png(outPath, outWidth, outHeight, 3, data, outWidth * 3);
	delete[] data;
	data = nullptr;
}

std::vector<std::pair<NNMatrix, NNMatrix>> batch;

// Load a the image from `imgPath` and create training batch data
void loadImage() {
	// Load the image data and metadata such as width and height (0 ignores the number of channels)
    unsigned char* data = stbi_load(imgPath, &width, &height, 0, 3); // 3 channels expected (RGB)
    if (data == NULL) {
        throw std::runtime_error("Error loading image: " + std::string(stbi_failure_reason()));
    }
	batch.resize(height * width);
	#pragma omp parallel for collapse(2) // Parallelize each iteration
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int idx = i * width + j;
			std::pair<NNMatrix, NNMatrix> sample;
			double y = static_cast<double>(i) / (height - 1) * 2 - 1;
			double x = static_cast<double>(j) / (width - 1) * 2 - 1;
			sample.first = NNMatrix::fromVector({y, x});
			sample.second.resize(3,1);
			for (int c = 0; c < 3; c++) {
				sample.second[c][0] = static_cast<double>(data[3 * idx + c]) / 255;
			}
			batch[idx] = sample;
		}
	}
    stbi_image_free(data);
}

// In this example, we will try to get the network to learn an image with an Implicit Neural Representation(INR)
// To do this, the image is first squished into a square with sides normalised to (-1, 1)
// Then, it approximates a function f(y,x) => {r,g,b}
// This script loads the image data, trains the network and creates a resulting image approximation
int main() {
	nn.setLayers({2,32,32,3});
	NNInitialization::xavierNormal(nn);
	nn.setActivationFunctions(NNActivationType::Tanh, NNActivationType::Sigmoid);
	nn.setLossFunction(NNLossType::MSE);

	// If there exists a data file `./nn.dat`, read from it
	std::ifstream in("./nn.dat", std::ios::binary);
	if (in.good()) nn.load(in);
	in.close();

	// Train the network with adam
	loadImage();
	NNTrainer trainer(nn, batch);
	trainer.learningRate = 0.1;
	trainer.epochCallback = []() { std::cout << "Epoch " << nn.epochsTrained << "\n"; };
	trainer.train(NNOptimizerType::Adam, 100);
	std::cout << "Training finished." << std::endl;
	createImage();

	// Write network data to `./nn.dat`
	std::ofstream out("./nn.dat", std::ios::binary);
	nn.save(out, true);
	out.close();
}
