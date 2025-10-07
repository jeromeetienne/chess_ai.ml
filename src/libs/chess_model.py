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

        # dropoutProbability = 0.2
        # conv1_out_channels = 64
        # conv2_out_channels = 128
        # conv3_out_channels = 16
        # fc_intermediate_size = 128

        conv1_out_channels = 32
        conv2_out_channels = 64
        conv3_out_channels = 128
        cls_fc_size = 128
        reg_fc_size = 64
        cls_dropoutProbability = 0.3
        reg_dropoutProbability = 0.2


        # dropoutProbability = 0.2
        # conv1_out_channels = 16
        # conv2_out_channels = 32
        # fc_intermediate_size = 256

        # dropoutProbability = 0.0
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
            nn.Conv2d(conv2_out_channels, conv3_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv3_out_channels),
            nn.ReLU(),
            torch.nn.Flatten(),
        )

        flat_features = conv3_out_channels * 8 * 8

        # classification head (move prediction)
        self.cls_head = nn.Sequential(
            nn.Linear(flat_features, cls_fc_size),
            nn.BatchNorm1d(cls_fc_size),
            nn.ReLU(),
            nn.Dropout(cls_dropoutProbability),
            nn.Linear(cls_fc_size, output_width),
        )
        
        # regression head (eval prediction)
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(flat_features, reg_fc_size),
            nn.BatchNorm1d(reg_fc_size),
            torch.nn.ReLU(),
            nn.Dropout(reg_dropoutProbability),
            torch.nn.Linear(reg_fc_size, 1),
        )
        
        # Initialize weights
        # nn.init.kaiming_uniform_(self.conv_1.weight, nonlinearity='leaky_relu')
        # nn.init.kaiming_uniform_(self.conv_2.weight, nonlinearity='leaky_relu')
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # x shape: (batch_size, 21, 8, 8)
        x = x.to(torch.float32)

        features = self.conv_layers(x)

        move_logits = self.cls_head(features)
        eval_pred = self.reg_head(features)

        return move_logits, eval_pred


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
        identity = x

        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)

        # Add the residual (skip connection)
        out += identity
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

        # res_block1_size = 64
        # res_block1_count = 2
        # res_block2_size = 128
        # res_block2_count = 2
        # cls_fc_size = 256
        # cls_fc_dropout = 0.2
        # reg_fc_size = 128
        # reg_fc_dropout = 0.2

        res_block1_size = 32
        res_block2_size = 64
        res_block1_count = 3
        res_block2_count = 3
        cls_fc_size = 128
        reg_fc_size = 64
        cls_fc_dropout = 0.2
        reg_fc_dropout = 0.2

        super().__init__()

        # Residual tower
        self.res_blocks1 = nn.Sequential(
            nn.Conv2d(input_channels, res_block1_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(res_block1_size),
            nn.ReLU(),
            *[ChessModelResNet_ResidualBlock(res_block1_size) for _ in range(res_block1_count)],
        )

        # Residual tower
        self.res_blocks2 = nn.Sequential(
            nn.Conv2d(res_block1_size, res_block2_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(res_block2_size),
            nn.ReLU(),
            *[ChessModelResNet_ResidualBlock(res_block2_size) for _ in range(res_block2_count)],
        )

        flat_features = res_block2_size * 8 * 8

        # classification head (move prediction)
        self.cls_head = nn.Sequential(
            # fully connected layers
            nn.Flatten(),
            nn.Linear(flat_features, cls_fc_size),
            nn.ReLU(),
            nn.Dropout(cls_fc_dropout),
            # Final classifier layer
            nn.Linear(cls_fc_size, output_width),
        )

        self.reg_head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(flat_features, reg_fc_size),
            nn.BatchNorm1d(reg_fc_size),
            torch.nn.ReLU(),
            nn.Dropout(reg_fc_dropout),
            torch.nn.Linear(reg_fc_size, 1),
        )

    def forward(self, x):
        # x: (batch, 21, 8, 8)
        features = x.to(torch.float32)

        features = self.res_blocks1(features)
        features = self.res_blocks2(features)

        move_logits = self.cls_head(features)
        eval_pred = self.reg_head(features)

        return move_logits, eval_pred

# =============================================================================
# AlphaZero exact model - NOT USED
# =============================================================================

class AlphaZeroNet_ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class AlphaZeroNet(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], output_shape: tuple[int]):
        super().__init__()

        in_channels, board_size, _ = input_shape
        action_size = output_shape[0]
        num_res_blocks = 10  # Number of residual blocks

        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.res_blocks = nn.ModuleList([AlphaZeroNet_ResidualBlock(256) for _ in range(num_res_blocks)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x: torch.Tensor):
        x = x.to(torch.float32)
        # Body
        x = F.relu(self.bn(self.conv(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = F.log_softmax(self.policy_fc(p), dim=1)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v
###############################################################################
###############################################################################
# 	 ChessModel class that wraps the original ChessModel
###############################################################################
###############################################################################
# class ChessModel(ChessModelConv2d):
class ChessModel(AlphaZeroNet):
    def __init__(self, input_shape: tuple[int, int, int], output_shape: tuple[int]):
        super(ChessModel, self).__init__(input_shape, output_shape)
