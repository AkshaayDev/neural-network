# neural-network

## Table of Contents

- [Overview](#overview)
- [Features](#features)
  1. [NNMatrix features](#1-nnmatrix-features)
  2. [Initializations](#2-initializations)
  3. [Activation functions](#3-activation-functions)
  4. [Loss functions](#4-loss-functions)
  5. [Optimizers](#5-optimizers)
- [Usage](#usage)
  1. [Cloning and Including](#1-cloning-and-including)
  2. [Creating and Configuring the Network](#2-creating-and-configuring-the-network)
  3. [Running the network](#3-running-the-network)
  4. [Saving and Loading](#4-saving-and-loading)
  5. [Training](#5-training)
- [Examples](#examples)

## Overview

This is a minimal C++ library for **creating simple neural networks** from scratch.
Apart from standard C++ libraries, it does not have any external dependencies.
This project is experimental and for educational purposes.

> Disclaimer: Although the neural network library headers do not have any external dependencies, some of the examples in the `/examples` folder may have external dependencies (like raylib.h or stb_image.h)

## Features

- Minimal purpose-built `NNMatrix` matrix class
- Feed-forward dense networks with `NeuralNetwork` class
- Network initialization, activation functions, loss functions
- Forward propagation and backpropagation
- Network saving and loading with a file stream
- Customizable trainer objects

### 1. NNMatrix features

- `std::vector<std::vector<double>>` constructor
- `rows` and `cols` constructor
- `rows()` and `cols()` getter functions
- Static matrix printing
- Static matching size checking
- Static `fromVector(std::vector<double>)` helper
- Static `fromScalar(double)` helper
- Resize rows and columns
- `forEach(const std::function<void(double*, int, int)>& func)` function to apply `func` to each element, passing element pointer, row and column as arguments
- `fill(double)` function to fill the matrix with the value
- Check for `nan`s
- Scalar and element-wise addition, subtraction, multiplication and division
- Scalar element-wise exponent
- Access data directly with `NNMatrix[row]`
- Static dot product
- Transpose of matrix
- Maximum value of matrix

### 2. Initializations

- Xavier (Normal/Uniform)
- He (Normal/Uniform)
- Constant biases

### 3. Activation functions

- Sigmoid
- ReLU
- tanh
- Softmax (output only)

### 4. Loss functions

- Mean Squared Error
- Categorical Cross Entropy

### 5. Optimizers

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

To set the density of each layer as well as network depth, use the `setLayers()` function.

```c++
nn.setLayers({2,2,2,1});
```

To initialize the parameters of the network, use one of the functions from the `NNInitialization` namespace.

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

To forward propagate an input, use `forwardPropagation()`.

```c++
// This performs forward propagation with the input and sets raw activations and activations
nn.forwardPropagation(input /*(NNMatrix)*/);
```

To run the network, use `run()`.

```c++
// This does not set anything and returns a NNMatrix output
nn.run(input /*(NNMatrix)*/);
```

To set partial derivatives, use `backwardPropagation()`.

```c++
nn.backwardPropagation(input /*(NNMatrix)*/, target /*(NNMatrix)*/);
```

### 4. Saving and Loading

To save network parameters and architecture, use the `save()` and `load()` functions.

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

To train the network, a trainer object must be created and initialized with the network and the batch.
The batch is a `std::vector` of samples which is a `std::pair` of the input and output `NNMatrix`.

```c++
std::vector<std::pair<NNMatrix, NNMatrix>> data;
// Example sample that maps {{0},{0}} to {{0}}
data.push_back(std::make_pair(NNMatrix::fromVector({0,0}), NNMatrix::fromScalar(0.0))); // 0 ^ 0 = 0

NNTrainer trainer(nn, batch);
```

> Note: The network and batch in the constructor are passed by reference
> If batch shuffling is enabled, it will modify the original batch too
> To prevent this, you can copy the batch into a new variable or disable shuffling (See below).

Callbacks for iterations and epochs can be set like this:

```c++
trainer.iterationCallback = []() { std::cout << "Iteration " << nn.iterationsTrained << "\n"; };
trainer.epochCallback = []() { std::cout << "Epoch " << nn.epochsTrained << "\n"; };
```

The batch is divided into samples to update parameters based on each sample's derivatives.
The sample size is `-1` by default, meaning the whole batch is trained in an iteration.
By default, the batch is shuffled before before every epoch but this can be disabled.

```c++
trainer.sampleSize = 128;
trainer.enableShuffling = false;
```

During training the optimizers use these hyperparameters by default:

- `learningRate` = 0.001 (used for gradient descent, momentum and adam)
- `beta` = 0.9 (used for momentum)
- `beta1` = 0.9 (used for adam)
- `beta2` = 0.999 (used for adam)
- `epsilon` = 1e-8 (used for adam)

These hyperparameters can be adjusted as trainer properties.

Finally call the `train()` method and pass a value from the `NNOptimizerType` enum class and the number of epochs.

```c++
trainer.train(NNOptimizerType::Adam, 100);
```

## Examples

- XOR Gate (`examples/xor/main.cpp`): Approximation of the boolean XOR gate
- Implicit Neural Representation (`examples/inr/main.cpp`): Recreation of an image
- MNIST digit classification (`examples/mnist/main.cpp`): Recognize handwritten digits
