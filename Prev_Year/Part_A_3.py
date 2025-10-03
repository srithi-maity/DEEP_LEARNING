import torch
import torch.nn as nn


class SingleLayerNN(nn.Module):
    def __init__(self, input_features=3, output=1):
        super(SingleLayerNN, self).__init__()
        self.linear = nn.Linear(input_features, output)

    def forward(self, x):
        return self.linear(x)


# Create model instance
model = SingleLayerNN(input_features=3, output=1)

# Generate random input
x = torch.randn(1, 3)  # batch_size=1, features=3
print(f"Input shape: {x.shape}")
print(f"Input values: {x}")

# Forward pass
output = model(x)
print(f"Output shape: {output.shape}")
print(f"Output value: {output.item()}")

# Print model parameters
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
    print(f"Values: {param.data}")