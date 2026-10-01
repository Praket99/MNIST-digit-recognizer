# MNIST Digit Recognizer From Scratch

A handwritten digit recognition neural network built from scratch using **Python and NumPy**.

The project trains a neural network on the **MNIST handwritten digit dataset** and provides a **Tkinter graphical interface** where you can draw a digit and let the neural network recognize it.

## Features

- Neural network implemented from scratch using NumPy
- No TensorFlow or PyTorch
- ReLU activation functions
- Softmax output layer
- Cross-entropy loss
- Backpropagation
- Stochastic Gradient Descent (SGD)
- He weight initialization
- MNIST dataset downloaded automatically
- Custom handwritten digit drawing interface
- Input preprocessing to make hand-drawn digits more similar to MNIST
- Model saving and loading
- 97.65% test accuracy on MNIST

## Neural Network Architecture

Input
  ↓
784 neurons
  ↓
128 neurons + ReLU
  ↓
128 neurons + ReLU
  ↓
10 neurons + Softmax
  ↓
Digit Prediction (0–9)