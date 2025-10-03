import numpy as np
import matplotlib.pyplot as plt

# Generate 100 equally spaced values between -10 and 10
z = np.linspace(-10, 10, 100)

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Tanh function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# ReLU function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Sigmoid
axes[0, 0].plot(z, sigmoid(z))
axes[0, 0].set_title('Sigmoid Function')
axes[0, 0].grid(True)

axes[0, 1].plot(z, sigmoid_derivative(z))
axes[0, 1].set_title('Sigmoid Derivative')
axes[0, 1].grid(True)

# Tanh
axes[1, 0].plot(z, tanh(z))
axes[1, 0].set_title('Tanh Function')
axes[1, 0].grid(True)

axes[1, 1].plot(z, tanh_derivative(z))
axes[1, 1].set_title('Tanh Derivative')
axes[1, 1].grid(True)

# ReLU
axes[2, 0].plot(z, relu(z))
axes[2, 0].set_title('ReLU Function')
axes[2, 0].grid(True)

axes[2, 1].plot(z, relu_derivative(z))
axes[2, 1].set_title('ReLU Derivative')
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()