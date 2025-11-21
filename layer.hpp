#ifndef LAYER_HPP
#define LAYER_HPP

#include "./neural-network.hpp"

class Layer {
public:
	int inCount = 0, outCount = 0; // Number of input and output neurons for the layer

	// Optional parameters and gradients for training
	std::vector<std::reference_wrapper<NNMatrix>> params;
	std::vector<NNMatrix> grads;
	// Optional last input and output storage for backpropagation
	NNMatrix lastInput, lastOutput;

	Layer(int inCount = 0, int outCount = 0) : inCount(inCount), outCount(outCount) {}

	virtual ~Layer() = default;
	// Returns an output without setting last input or output
	virtual NNMatrix run(const NNMatrix& x) = 0;
	// Returns an output and sets last input and/or output
	virtual NNMatrix forward(const NNMatrix& x) = 0;
	// Sets gradients and returns error for input
	virtual NNMatrix backward(const NNMatrix& dy) = 0;

	// Save layer data to the file stream
	virtual void save(std::ofstream& out) = 0;
	// Factory loader
	static std::unique_ptr<Layer> load(std::ifstream& in);
};

class ActivationLayer : public Layer {
public:
	std::string fnName;
	std::function<NNMatrix(NNMatrix)> f, g;
	ActivationLayer(int count, std::string fnName) : Layer(count, count), fnName(fnName) {
		if (fnName == NNActivationType::Sigmoid) {
			f = NNActivation::sigmoid;
			g = [this](NNMatrix dy) { return NNActivation::sigmoidDerivative(lastOutput) * dy; };
		} else if (fnName == NNActivationType::ReLU) {
			f = NNActivation::relu;
			g = [this](NNMatrix dy) { return NNActivation::reluDerivative(lastOutput) * dy; };
		} else if (fnName == NNActivationType::Tanh) {
			f = NNActivation::tanh;
			g = [this](NNMatrix dy) { return NNActivation::tanhDerivative(lastOutput) * dy; };		
		} else if (fnName == NNActivationType::Softmax) {
			f = NNActivation::softmax;
			g = [this](NNMatrix dy) { return NNActivation::softmaxDerivative(lastOutput, dy); };
		} else throw std::runtime_error("Unknown hidden activation function ('" + fnName + "')");
	}

	NNMatrix run(const NNMatrix& x) override { return f(x); }
	NNMatrix forward(const NNMatrix& x) override { lastOutput = f(x); return lastOutput; }
	NNMatrix backward(const NNMatrix& dy) override { return g(dy); }

	void save(std::ofstream& out) override {
		// Write the layer type
		const std::string type = "Activation";
		uint32_t size = type.size();
		out.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
		out.write(type.c_str(), size);
		// Write the number of neurons
		out.write(reinterpret_cast<const char*>(&inCount), sizeof(int));
		// Write the activation function name
		size = fnName.size();
		out.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
		out.write(fnName.c_str(), size);
	}
	static std::unique_ptr<ActivationLayer> load(std::ifstream& in) {
		// Layer type was read by static Layer::load
		// Read the number of neurons
		int count;
		in.read(reinterpret_cast<char*>(&count), sizeof(int));
		// Read the activation function
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
		std::string fnName;
		fnName.resize(size);
		in.read(&fnName[0], size);
		return std::make_unique<ActivationLayer>(count, fnName);
	}
};

class DenseLayer : public Layer {
public:
	NNMatrix W, B;
	DenseLayer(int in, int out) : Layer(in, out) {
		W.resize(out, in);
		B.resize(out, 1);
		params = { std::ref(W), std::ref(B) };
		grads.resize(2);
		grads[0].resize(out, in);
		grads[1].resize(out, 1);
	}

	NNMatrix run(const NNMatrix& x) override { return NNMatrix::dot(W, x) + B; } // y = W . x + B
	NNMatrix forward(const NNMatrix& x) override { lastInput = x; return run(x); }
	NNMatrix backward(const NNMatrix& dy) override {
		grads[0] = NNMatrix::dot(dy, lastInput.transpose()); // dW = dy . x^T
		grads[1] = dy; // dB = dy
		return NNMatrix::dot(W.transpose(), dy); // dx = W^T . dy
	}

	void save(std::ofstream& out) override {
		// Write the layer type
		const std::string type = "Dense";
		uint32_t size = type.size();
		out.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
		out.write(type.c_str(), size);
		// Write the number of input and output neurons
		out.write(reinterpret_cast<const char*>(&inCount), sizeof(int));
		out.write(reinterpret_cast<const char*>(&outCount), sizeof(int));
		// Write the weights and biases
		for (NNMatrix& param : params) {
			param.forEach([&out](double *val, int, int) {
				out.write(reinterpret_cast<const char*>(val), sizeof(double));
			});
		}
	}
	static std::unique_ptr<DenseLayer> load(std::ifstream& in) {
		// Layer type was read by static Layer::load
		// Read the number of input and output neurons
		int inCount, outCount;
		in.read(reinterpret_cast<char*>(&inCount), sizeof(int));
		in.read(reinterpret_cast<char*>(&outCount), sizeof(int));
		std::unique_ptr<DenseLayer> layer = std::make_unique<DenseLayer>(inCount, outCount);
		// Read the weights and biases
		for (NNMatrix& mat : layer->params) {
			for (int i = 0; i < mat.rows(); i++) {
				in.read(reinterpret_cast<char*>(mat[i].data()), mat.cols() * sizeof(double));
			}
		}
		return layer;
	}
};

std::unique_ptr<Layer> Layer::load(std::ifstream& in) {
	std::string type;
	uint32_t size = 0;
	in.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
	type.resize(size);
	in.read(&type[0], size);
	if (type == "Activation") return ActivationLayer::load(in);
	if (type == "Dense") return DenseLayer::load(in);
	throw std::runtime_error("Unknown layer type found.");
}

#endif
