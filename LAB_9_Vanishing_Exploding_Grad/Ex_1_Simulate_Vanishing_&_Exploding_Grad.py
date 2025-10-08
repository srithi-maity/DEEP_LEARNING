import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

torch.manual_seed(42)
np.random.seed(42)

# Make data: Two circles on x-y plane as a classification problem
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)


class DeepMLP(nn.Module):
    def __init__(self, activation, n_hidden=10, n_layers=10):
        super().__init__()
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()
        }
        layers = [nn.Linear(2, n_hidden)]
        layers.append(activations[activation])
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(activations[activation])
        layers.append(nn.Linear(n_hidden, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_and_monitor_gradients(X, y, activation="sigmoid", n_layers=10, n_epochs=100, batch_size=32):
    model = DeepMLP(activation=activation, n_layers=n_layers)
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters())

    # For listing gradients
    gradient_history = {f'layer_{i}': [] for i in range(n_layers)}
    loss_history = []

    print(f"Training with {activation} activation and {n_layers} layers...")

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(X.size(0))

        # Use only first batch for gradient monitoring
        idx = perm[:batch_size]
        X_batch, y_batch = X[idx], y[idx]

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()

        # gradients for each layer
        for i, (name, param) in enumerate(model.named_parameters()):
            if 'weight' in name and param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                gradient_history[f'layer_{i // 2}'].append(grad_norm)  # i//2 because each layer has weight and bias

        optimizer.step()
        loss_history.append(loss.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return gradient_history, loss_history


def plot_gradient_analysis(gradient_history, loss_history, activation):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Gradient norms per layer (linear scale)
    for layer, gradients in gradient_history.items():
        axes[0, 0].plot(gradients, label=layer)
    axes[0, 0].set_title(f'Gradient Norms per Layer ({activation}) - Linear Scale')
    axes[0, 0].set_ylabel('Gradient Norm')
    axes[0, 0].legend()

    # Plot 2: Gradient norms per layer (log scale)
    for layer, gradients in gradient_history.items():
        axes[0, 1].semilogy(gradients, label=layer)
    axes[0, 1].set_title(f'Gradient Norms per Layer ({activation}) - Log Scale')
    axes[0, 1].set_ylabel('Gradient Norm (log)')
    axes[0, 1].legend()

    # Plot 3: Final gradient distribution across layers
    final_gradients = [grads[-1] for grads in gradient_history.values()]
    layers = list(gradient_history.keys())
    axes[1, 0].bar(layers, final_gradients)
    axes[1, 0].set_title(f'Final Gradient Norms by Layer ({activation})')
    axes[1, 0].set_ylabel('Final Gradient Norm')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Plot 4: Loss curve
    axes[1, 1].plot(loss_history)
    axes[1, 1].set_title(f'Training Loss ({activation})')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()


# Testing with different activations with deep network
n_layers = 10  # Deep network to demonstrate vanishing/exploding gradients

for activation in ["sigmoid", "tanh", "relu"]:
    gradient_history, loss_history = train_and_monitor_gradients(X, y, activation=activation, n_layers=n_layers, n_epochs=100)
    plot_gradient_analysis(gradient_history, loss_history, activation)