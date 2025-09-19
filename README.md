# neural-network

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
  1. [Cloning and Including](#1-cloning-and-including)
  2. [Creating and Configuring the Network](#2-creating-and-configuring-the-network)
  3. [Running the network](#3-running-the-network)
  4. [Saving and Loading](#4-saving-and-loading)
  5. [Training](#5-training)

## Overview
This is a minimal C++ library for **creating simple neural networks** from scratch.
This project is experimental and for educational purposes.

## Features
- Minimal purpose-built `NNMatrix` matrix class
- Feed-forward dense networks with `NeuralNetwork` class
- Network initialization, activation functions, loss functions
- Backpropagation and network training
- Network saving and loading with a file stream

`NNMatrix` features:
- `std::vector<std::vector<double>>` constructor
- `rows` and `cols` constructor
- `rows()` and `cols()` getter functions
- Static matrix printing
- Static `fromVector(std::vector<double>)` helper
- Static `fromScalar(double)` helper
- Resize rows and columns
- `forEach(const std::function<void(double*, int, int)>& func)` function to apply `func` to each element, passing element pointer, row and column as arguments
- `fill(double)` function to fill the matrix with the value
- Scalar and element-wise addition, subtraction, multiplication and division
- Scalar element-wise exponent
- Access data directly with `NNMatrix[row]`
- Static dot product
- Transpose of matrix

Initializations:
- Xavier (Normal/Uniform)
- He (Normal/Uniform)
- Biases to constant
- Biases to 0

Activation functions:
- Sigmoid
- ReLU
- tanh

Loss functions:
- MSE

Trainers:
- Gradient Descent
- Momentum
- Adam

## Usage

### 1. Cloning and Including
Clone this repository:
```bash
git clone https://github.com/akshaaydev/neural-network.git
```

Then, include from the main neural network header `neural-network.hpp`.
```c++
#include "/neural-network/neural-network.hpp"
```

### 2. Creating and Configuring the Network
To create a network, define a `NeuralNetwork` object.
```c++
NeuralNetwork nn;
```

To set the density of each layer as well as network depth, use the `setLayers()` function
```c++
nn.setLayers({2,2,2,1});
```

To initialize the parameters of the network, use one of the functions from the `NNInitialization` namespace
```c++
NNInitialization::xavierNormal(nn);
```

To set the hidden and output activation functions, use these functions:
```c++
// This sets hidden layer activations to sigmoid and output layer activations to sigmoid
nn.setActivationFunctions(NNActivationType::Sigmoid, NNActivationType::Sigmoid);
// This sets the loss function to Mean Squared Error(MSE)
nn.setLossFunction(NNLossType::MSE);
```

### 3. Running the Network
To forward propagate an input, use `forwardPropagation()`
```c++
// This performs forward propagation with the input and sets raw activations and activations
nn.forwardPropagation(input /*(NNMatrix)*/);
```

To run the network, use `run()`
```c++
// This does not set anything and returns a NNMatrix output
nn.run(input /*(NNMatrix)*/);
```

To set partial derivatives, use `backwardPropagation()`
```c++
nn.backwardPropagation(input /*(NNMatrix)*/, target /*(NNMatrix)*/);
```

### 4. Saving and Loading
To save network parameters and architecture, use the `save()` and `load()` functions
```c++
// Write the neural network data to a file `out.dat` with binary encoding
std::ofstream out("out.dat", std::ios::binary);
nn.save(out); // Save the network data to that file
out.close(); // Close the file
```
```c++
// Read the neural network data from a file `in.dat` with binary encoding
std::ifstream in("in.dat", std::ios::binary);
nn.load(in); // Load the network data from that file
in.close(); // Close the file
```

### 5. Training
To train the network, call a function from the `NNTrainer` namespace and pass the network, training data, hyperparameters (depending on training method) and a callback function to be called after each epoch
```c++
// Data is a std::vector<std::pair<NNMatrix, NNMatrix>>
// Each sample from it is a pair of the input NNMatrix and the expected output NNMatrix
NNTrainer::gradientDescent(nn /*(NeuralNetwork&)*/, data, {
	{ "learning_rate", 10 },
	{ "iterations", 1000 }
}, [](){});
```

> Refer to `examples/xor/main.cpp` for a sample usage to approximate an XOR gate.
