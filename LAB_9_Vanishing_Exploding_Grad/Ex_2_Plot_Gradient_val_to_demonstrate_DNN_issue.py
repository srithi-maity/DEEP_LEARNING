import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def create_very_deep_network(activation_fn, depth=20, hidden_size=10):
    """Create a very deep network to demonstrate vanishing gradients"""
    activations = {
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "relu": nn.ReLU()
    }

    layers = [nn.Linear(2, hidden_size)]
    layers.append(activations[activation_fn])

    for i in range(depth - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activations[activation_fn])

    layers.append(nn.Linear(hidden_size, 1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def analyze_vanishing_gradient():
    torch.manual_seed(42)

    # Create sample input
    x = torch.randn(1, 2, requires_grad=True)
    target = torch.tensor([[1.0]])

    depths = [5, 10, 15, 20, 25, 30]
    activations = ["sigmoid", "tanh", "relu"]

    results = {act: [] for act in activations}

    for depth in depths:
        print(f"Testing depth: {depth}")
        for activation in activations:
            model = create_very_deep_network(activation, depth=depth)

            # Forward pass
            output = model(x)

            # Compute loss and backward pass
            loss = nn.BCELoss()(output, target)
            loss.backward()

            # Calculate average gradient norm for all parameters
            total_grad_norm = 0
            param_count = 0

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    total_grad_norm += grad_norm
                    param_count += 1

            avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
            results[activation].append(avg_grad_norm)

            # Clean up for next iteration
            model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()

    return depths, results


def plot_vanishing_gradient_analysis(depths, results):
    plt.figure(figsize=(12, 8))

    # Plot 1: Gradient norms vs network depth
    plt.subplot(2, 2, 1)
    for activation, gradients in results.items():
        plt.semilogy(depths, gradients, 'o-', label=activation, linewidth=2, markersize=8)
    plt.xlabel('Network Depth (Number of Layers)')
    plt.ylabel('Average Gradient Norm (log scale)')
    plt.title('Vanishing Gradient: Gradients vs Network Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Relative gradient decay
    plt.subplot(2, 2, 2)
    for activation, gradients in results.items():
        relative_gradients = [g / gradients[0] for g in gradients]  # Normalize to first value
        plt.plot(depths, relative_gradients, 'o-', label=activation, linewidth=2, markersize=8)
    plt.xlabel('Network Depth (Number of Layers)')
    plt.ylabel('Relative Gradient Magnitude')
    plt.title('Relative Gradient Decay')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Gradient ratios between consecutive layers
    plt.subplot(2, 2, 3)
    for activation, gradients in results.items():
        ratios = []
        for i in range(1, len(gradients)):
            if gradients[i - 1] > 0:
                ratio = gradients[i] / gradients[i - 1]
                ratios.append(ratio)
        plt.plot(depths[1:], ratios, 'o-', label=activation, linewidth=2, markersize=8)
    plt.xlabel('Network Depth')
    plt.ylabel('Gradient Ratio (layer_n / layer_{n-1})')
    plt.title('Gradient Propagation Ratios')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Theoretical vs observed
    plt.subplot(2, 2, 4)
    # Theoretical bounds for different activations
    x_theoretical = np.array(depths)

    # Sigmoid: derivative max is 0.25
    sigmoid_theoretical = 0.25 ** (x_theoretical - 1)
    plt.semilogy(x_theoretical, sigmoid_theoretical, '--', label='Sigmoid Theoretical', alpha=0.7)

    # Tanh: derivative max is 1.0, but typically <1
    tanh_theoretical = 0.8 ** (x_theoretical - 1)  # Approximation
    plt.semilogy(x_theoretical, tanh_theoretical, '--', label='Tanh Theoretical', alpha=0.7)

    # ReLU: derivative is 1 for positive inputs
    relu_theoretical = 0.5 ** (x_theoretical - 1)  # Approximation considering dead neurons
    plt.semilogy(x_theoretical, relu_theoretical, '--', label='ReLU Theoretical', alpha=0.7)

    plt.xlabel('Network Depth')
    plt.ylabel('Theoretical Gradient Bound')
    plt.title('Theoretical Gradient Bounds')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Run the analysis
print("Analyzing vanishing gradient problem...")
depths, results = analyze_vanishing_gradient()
plot_vanishing_gradient_analysis(depths, results)

# Print summary
print("\n=== Vanishing Gradient Analysis Summary ===")
for activation in results:
    print(f"\n{activation.upper()}:")
    for depth, grad in zip(depths, results[activation]):
        print(f"  Depth {depth}: Gradient norm = {grad:.6f}")

    # Calculate decay rate
    if len(results[activation]) > 1:
        decay_rate = results[activation][-1] / results[activation][0]
        print(f"  Total decay over {depths[-1]} layers: {decay_rate:.2e}")