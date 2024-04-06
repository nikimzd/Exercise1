# Feed Forward Neural Network

This repository contains a simple implementation of a feed-forward neural network in Python using NumPy for binary classification tasks.

## Overview

The `FeedForwardNN` class is a neural network implementation with the following features:

- Initialization of weights and biases for the input, hidden, and output layers
- Forward propagation using the sigmoid activation function
- Backpropagation to compute gradients and update weights and biases
- Training loop with configurable epochs and learning rate
- Prediction function to evaluate the model on new data

The code also includes a function `generate_data` to create synthetic binary classification data for training and testing purposes.

## Usage

1. Clone the repository or copy the code into a Python file.
2. Run the code, which will:
   - Generate synthetic training and testing data
   - Create and train a neural network with specified hyperparameters
   - Evaluate the model's accuracy on the test data
   - Visualize the decision boundary using Matplotlib

You can modify the hyperparameters (`input_size`, `hidden_size`, `output_size`, `epochs`, `learning_rate`) and the data generation function (`generate_data`) to experiment with different scenarios.

## Dependencies

- NumPy
- Matplotlib

Make sure you have these libraries installed before running the code.
