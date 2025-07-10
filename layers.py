import numpy as np
from optimizers import Optimizer

class Layer:
    def __init__(self):
        pass

    def forward(self, input):
        raise NotImplementedError("Forward pass not implemented.")

    def backward(self, grad_output):
        raise NotImplementedError("Backward pass not implemented.")

    def update(self, optimizer: Optimizer):
        pass  # Optional to override


class Dense(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.biases = np.zeros((1, output_dim))
        self.input = None
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, grad_output):
        self.grad_weights = np.dot(self.input.T, grad_output)
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.weights.T)

    def update(self, optimizer: Optimizer):
        optimizer.update(self)


class Activation(Layer):
    def __init__(self, activation):
        self.activation_name = activation.lower()
        self.input = None

    def forward(self, input):
        self.input = input
        if self.activation_name == "relu":
            return np.maximum(0, input)
        elif self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-input))
        elif self.activation_name == "softmax":
            exp = np.exp(input - np.max(input, axis=1, keepdims=True))  # stability
            return exp / np.sum(exp, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

    def backward(self, grad_output):
        if self.activation_name == "relu":
            return grad_output * (self.input > 0)
        elif self.activation_name == "sigmoid":
            sig = 1 / (1 + np.exp(-self.input))
            return grad_output * sig * (1 - sig)
        elif self.activation_name == "softmax":
            # Assumes use with cross-entropy loss, so dL/dz = softmax - y_true
            return grad_output
