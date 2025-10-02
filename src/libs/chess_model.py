# pip imports
import torch.nn as nn
import torch
import torch.nn.functional as F


class ChessModelOriginal(nn.Module):
    def __init__(self, num_classes):
        super(ChessModelOriginal, self).__init__()
        # conv1 -> relu -> conv2 -> relu -> flatten -> fc1 -> relu -> fc2
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

        # Initialize weights
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # x shape: (batch_size, 14, 8, 8)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits
        return x


###############################################################################
#   Improved ChessModel with more conv layers and dropout
#
class ChessModelConv2d(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], output_shape: tuple[int]):
        super(ChessModelConv2d, self).__init__()

        output_width = output_shape[0]
        input_channels, input_height, input_width = input_shape

        dropoutProbability = 0.2
        conv1_out_channels = 64
        conv2_out_channels = 128
        # conv3_out_channels = 128
        # conv4_out_channels = 128
        fc_intermediate_size = 1024

        # dropoutProbability = 0.2
        # conv1_out_channels = 16
        # conv2_out_channels = 32
        # fc_intermediate_size = 256

        # dropoutProbability = 0.2
        # conv1_out_channels = 8
        # conv2_out_channels = 16
        # fc_intermediate_size = 256

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, conv1_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv1_out_channels),
            nn.ReLU(),
            # nn.Dropout2d(dropoutProbability),
            nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2_out_channels),
            nn.ReLU(),
            # nn.Dropout2d(dropoutProbability),
            # nn.Conv2d(conv2_out_channels, conv3_out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(conv3_out_channels),
            # nn.ReLU(),
            # # nn.Dropout2d(dropoutProbability),
            # nn.Conv2d(conv3_out_channels, conv4_out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(conv4_out_channels),
            # nn.ReLU(),
            # # nn.Dropout2d(dropoutProbability),
        )

        # self.gap_2d = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.flatten = nn.Flatten()

        self.fc_layers = nn.Sequential(
            nn.Linear(conv2_out_channels * 8 * 8, fc_intermediate_size),
            nn.BatchNorm1d(fc_intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropoutProbability),
            nn.Linear(fc_intermediate_size, output_width),
        )

        # Initialize weights
        # nn.init.kaiming_uniform_(self.conv_1.weight, nonlinearity='leaky_relu')
        # nn.init.kaiming_uniform_(self.conv_2.weight, nonlinearity='leaky_relu')
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # x shape: (batch_size, 21, 8, 8)
        x = x.to(torch.float32)

        x = self.conv_layers(x)

        # x = self.gap_2d(x)
        x = self.flatten(x)

        x = self.fc_layers(x)

        return x


###############################################################################
#   Residual Block and ChessMoveResNet for improved performance
#


class ChessModelResNet_ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Store the original input for the skip connection
        residual = x

        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)

        # Add the residual (skip connection)
        out += residual
        out = F.relu(out)

        return out


class ChessModelResNet(nn.Module):
    """
    A ResNet-like model for classifying chess board states.
    - Input: A tensor of shape (N, 21, 8, 8) where N is the batch size.
    - Output: A tensor of logits of shape (N, 1968).
    """

    def __init__(self, input_shape: tuple[int, int, int], output_shape: tuple[int]):

        output_width = output_shape[0]
        input_channels, input_height, input_width = input_shape
        resblock_count = 8
        resblock_channels = 64
        # resblock_channels = 128
        fc_intermediate_size = 512
        dropoutProbability = 0.2

        super().__init__()

        # Input convolution (expand feature space)
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, resblock_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(resblock_channels), nn.ReLU()
        )

        # Residual tower
        self.res_blocks = nn.Sequential(
            # Add multiple residual blocks
            *[ChessModelResNet_ResidualBlock(resblock_channels) for _ in range(resblock_count)],
            # a final conv layer to reduce channels before the classifier
            nn.Conv2d(resblock_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # classifier
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, fc_intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropoutProbability),
            # Final classifier layer
            nn.Linear(fc_intermediate_size, output_width),
        )

    def forward(self, x):
        # x: (batch, 21, 8, 8)
        x = x.to(torch.float32)

        # Initial processing
        x = self.input_conv(x)

        # Pass through residual blocks
        x = self.res_blocks(x)

        # Final classification layer
        x = self.fc(x)
        return x


###############################################################################
###############################################################################
# 	 ChessModel class that wraps the original ChessModel
###############################################################################
###############################################################################
class ChessModel(ChessModelConv2d):
    def __init__(self, input_shape: tuple[int, int, int], output_shape: tuple[int]):
        super(ChessModel, self).__init__(input_shape, output_shape)
