# Neural Network

A simple implementation of a neural network in Python using NumPy.

## Overview

This repository contains a basic neural network implementation with the following features:

- Feedforward neural network with one hidden layer
- Sigmoid activation function
- Backpropagation learning algorithm
- Example of solving the XOR problem

## Files

- `simple_nn.py`: The neural network class implementation
- `example.py`: An example using the neural network to solve the XOR problem

## Usage

To run the example:

```bash
python example.py
```

### Requirements

- NumPy

### Example Output

```
Training the neural network...
Epoch 0, Loss: 0.25001646331648344
Epoch 1000, Loss: 0.2473143935563291
Epoch 2000, Loss: 0.24450188862941114
Epoch 3000, Loss: 0.2415324608586338
Epoch 4000, Loss: 0.23833962262631
Epoch 5000, Loss: 0.23484518953442596
Epoch 6000, Loss: 0.2309495057063669
Epoch 7000, Loss: 0.22654733499798536
Epoch 8000, Loss: 0.22153044396493248
Epoch 9000, Loss: 0.2157693574131686

Testing the neural network:
Input: [0 0], Target: [0], Prediction: 0.0843
Input: [0 1], Target: [1], Prediction: 0.9069
Input: [1 0], Target: [1], Prediction: 0.9068
Input: [1 1], Target: [0], Prediction: 0.0941
```

## Future Improvements

- Add more activation functions (ReLU, tanh, etc.)
- Implement more layers
- Add regularization techniques
- Support for different loss functions
- Batch processing
