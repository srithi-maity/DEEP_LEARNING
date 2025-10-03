import torch
import torch.nn as nn


class DoubleLayerNN(nn.Module):
    def __init__(self, input_features=3, hidden_features=5, output=1):
        super(DoubleLayerNN, self).__init__()
        # First layer: Input to Hidden
        self.layer1 = nn.Linear(input_features, hidden_features)
        # Activation function
        self.activation = nn.ReLU()
        # Second layer: Hidden to Output
        self.layer2 = nn.Linear(hidden_features, output)

    def forward(self, x):
        # First layer forward pass
        hidden_output = self.layer1(x)
        # Apply activation function
        activated_output = self.activation(hidden_output)
        # Second layer forward pass
        final_output = self.layer2(activated_output)
        return final_output


# Create model instance
model = DoubleLayerNN(input_features=3, hidden_features=5, output=1)

# Generate random input
x = torch.randn(1, 3)  # batch_size=1, features=3
print(f"Input shape: {x.shape}")
print(f"Input values: {x}")

# Forward pass
output = model(x)
print(f"Output shape: {output.shape}")
print(f"Output value: {output.item()}")

# Print model architecture
print("\nModel Architecture:")
print(f"Input Layer: 3 neurons")
print(f"Hidden Layer: 5 neurons with ReLU activation")
print(f"Output Layer: 1 neuron")
print(model)

# Print model parameters
print("\nModel Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
    print(f"Values: {param.data}")