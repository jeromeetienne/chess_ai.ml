# stdlib imports
import math

# pip imports
import torch.nn as nn
import torch


class WeightInitializer:
    @staticmethod
    def init_weights(model: nn.Module) -> None:
        """
        Initializes the weights of a model module based on its type
        The goal is to set the model's parameters (weights and biases) to a sensible starting point before it sees any data.

        Usage:
            model = YourModel()
            model.apply(init_weights)

        ## Initialization Strategies
        * **Conv / Linear Layers (`nn.init.kaiming_normal_`)**: This is the **He initialization**. It's designed specifically for layers that are followed by a
        Rectified Linear Unit (ReLU) activation, which is the most common case in modern neural networks. It helps prevent gradients from vanishing or exploding
        by maintaing the variance of the activations.
        * **BatchNorm / GroupNorm (`nn.init.constant_`)**: These layers are initialized with $\gamma$ (weight) = 1 and $\beta$ (bias) = 0. This ensures that at
        the very beginning of training, the batch/group norm layer simply passes the normalized input through, acting as an identity transformation. The network
        can then learn the optimal scaling and shifting.
        * **RNN / LSTM / GRU (`nn.init.xavier_uniform_`)**: These layers often use **Xavier (or Glorot) initialization**. It's a good general-purpose initializer
        that works well with activations like `tanh` or `sigmoid`, which were historically common in RNNs.
        * **LSTM Forget Gate (Bias = 1)**: This is a very common trick. By setting the forget gate's bias to a positive value (like 1), the gate's output
        (after the sigmoid) will be close to 1. This encourages the LSTM cell to *remember* its previous state by default, which is a better starting point
        than forgetting everything.

        """

        model.apply(WeightInitializer._init_weights_module)

    @staticmethod
    def _init_weights_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            # Kaiming (He) Normal initialization for layers with ReLU non-linearity
            # This is the default and most common case for modern architectures.
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                # Initialize bias to zero
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            # Initialize BatchNorm/GroupNorm weights to 1 and bias to 0
            # This makes the layer an identity transformation at the start of training.
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

        elif isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            # Initialize recurrent layers
            for name, param in module.named_parameters():
                if "weight_ih" in name or "weight_hh" in name:
                    # Xavier (Glorot) Uniform initialization for weight matrices
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    # Initialize bias to zero
                    nn.init.constant_(param, 0)

            # Special case for LSTM: Initialize forget gate bias to 1
            # This helps the model to remember information by default at the beginning of training.
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "bias" in name:
                        # Biases are concatenated (input, forget, cell, output)
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.0)

        elif isinstance(module, nn.Embedding):
            # Initialize embedding layers with a normal distribution
            nn.init.normal_(module.weight, mean=0, std=0.02)


# =============================================================================
# Mini test in __main__
# =============================================================================


def main():
    # 1. Define your model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 16 * 16, 120)  # Assuming 32x32 input
            self.fc2 = nn.Linear(120, 10)

        def forward(self, x):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 2. Instantiate your model
    model = SimpleCNN()

    # 3. Apply the weight initialization function
    WeightInitializer.init_weights(model)

    print("Model weights initialized successfully.")

    # You can check the initialization
    print("Conv1 weight (first 5 values):", model.conv1.weight.data.flatten()[:5])
    print(model.conv1.weight.data.flatten()[:5])
    print(f"Conv1 bias: {model.conv1.bias.data}")  # type: ignore

    print(f"BatchNorm1 weight: {model.bn1.weight.data}")
    print(f"BatchNorm1 bias: {model.bn1.bias.data}")

    print("FC1 weight (first 5 values):")
    print(model.fc1.weight.data.flatten()[:5])


if __name__ == "__main__":
    main()
