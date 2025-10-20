# pip imports
import dataclasses
import torch.nn as nn
import torch
import torch.nn.functional as F


# =============================================================================
# Base classes
# =============================================================================
class ChessModelParams:
    """
    Base class for chess model parameters.
    All chess model parameter classes should inherit from this class.
    """

    pass


class ChessModel(nn.Module):
    """
    Base class for chess models.
    All chess models should inherit from this class.
    """

    pass


# =============================================================================
# Original Chess Model
# =============================================================================


class ChessModelOriginal(ChessModel):
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


# =============================================================================
# =============================================================================
# ChessModelConv2d
# =============================================================================
# =============================================================================


@dataclasses.dataclass
class ChessModelConv2dParams(ChessModelParams):
    """
    Parameters for the ChessModelConv2d model.
    """

    conv_out_channels: list[int]
    """ List of output channels for each convolutional layer. """
    cls_fc_size: int
    """ Size of the fully connected layer in the classification head. """
    reg_fc_size: int
    """ Size of the fully connected layer in the regression head. """
    cls_head_dropout: float
    """ Dropout probability for the classification head. """
    reg_head_dropout: float
    """ Dropout probability for the regression head. """

    # conv_out_channels: list[int] = dataclasses.field(default_factory=lambda: [32, 64, 128])
    # """ List of output channels for each convolutional layer. """
    # cls_fc_size: int = 128
    # """ Size of the fully connected layer in the classification head. """
    # reg_fc_size: int = 64
    # """ Size of the fully connected layer in the regression head. """
    # cls_head_dropout: float = 0.2
    # """ Dropout probability for the classification head. """
    # reg_head_dropout: float = 0.2
    # """ Dropout probability for the regression head. """


class ChessModelConv2d(ChessModel):
    PROFILE: dict[str, ChessModelConv2dParams] = {
        "fast": ChessModelConv2dParams(
            conv_out_channels=[16, 32, 64],
            cls_fc_size=64,
            reg_fc_size=64,
            cls_head_dropout=0.0,
            reg_head_dropout=0.0,
        ),
        "default": ChessModelConv2dParams(
            conv_out_channels=[64, 128, 256],
            cls_fc_size=128,
            reg_fc_size=64,
            cls_head_dropout=0.5,
            reg_head_dropout=0.5,
        ),
        "slow": ChessModelConv2dParams(
            conv_out_channels=[16, 64, 128, 256],
            cls_fc_size=256,
            reg_fc_size=128,
            cls_head_dropout=0.3,
            reg_head_dropout=0.3,
        ),
        "optimized": ChessModelConv2dParams(
            conv_out_channels=[16, 32, 64],
            cls_fc_size=256,
            reg_fc_size=32,
            cls_head_dropout=0.1,
            reg_head_dropout=0.2,
        ),
    }

    def __init__(self, input_shape: tuple[int, int, int], output_shape: tuple[int], params: ChessModelParams):
        """
        A Convolutional Neural Network (CNN) model for classifying chess board states.

        Args:
            input_shape (tuple): A tuple representing the shape of the input tensor (channels, height, width).
            output_shape (tuple): A tuple representing the shape of the output tensor (number of classes,).
            params (ChessModelConv2dParams): Hyperparameters for the model architecture.
        """

        super(ChessModelConv2d, self).__init__()

        # =============================================================================
        # Handle arguments
        # =============================================================================

        output_width = output_shape[0]
        input_channels, input_height, input_width = input_shape

        # Validate and convert parameters
        # - Sometime we might get a base class, so we need get the default of the base class as a placeholder
        assert isinstance(params, ChessModelParams), "params must be an instance of ChessModelParams"
        conv2d_params: ChessModelConv2dParams = params if isinstance(params, ChessModelConv2dParams) else ChessModelConv2d.PROFILE["default"]

        # =============================================================================
        # Build the conv_layers
        # =============================================================================

        conv_channels_out = conv2d_params.conv_out_channels
        conv_layers: list[nn.Module] = []
        for layer_idx in range(len(conv_channels_out)):
            in_channels = input_channels if layer_idx == 0 else conv_channels_out[layer_idx - 1]
            out_channels = conv_channels_out[layer_idx]
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())

        self.conv_layers = nn.Sequential(
            *conv_layers,
            torch.nn.Flatten(),
        )

        # =============================================================================
        # Build the classification head (move prediction))
        # =============================================================================

        flat_features = conv_channels_out[-1] * 8 * 8
        self.cls_head = nn.Sequential(
            nn.Linear(flat_features, conv2d_params.cls_fc_size),
            nn.BatchNorm1d(conv2d_params.cls_fc_size),
            nn.ReLU(),
            nn.Dropout(conv2d_params.cls_head_dropout),
            nn.Linear(conv2d_params.cls_fc_size, output_width),
        )

        # =============================================================================
        # Build the regression head (eval prediction)
        # =============================================================================

        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(flat_features, conv2d_params.reg_fc_size),
            nn.BatchNorm1d(conv2d_params.reg_fc_size),
            torch.nn.ReLU(),
            nn.Dropout(conv2d_params.reg_head_dropout),
            torch.nn.Linear(conv2d_params.reg_fc_size, 1),
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


@dataclasses.dataclass
class ChessModelResnetParams(ChessModelParams):
    """
    Parameters for the ChessModelResNet model.
    """

    res_block_sizes: list[int]
    """ List of output channels for each residual block layer. """
    res_block_counts: list[int]
    """ List of number of residual blocks for each layer. """
    cls_head_size: int
    """ Size of the fully connected layer in the classification head. """
    reg_head_size: int
    """ Size of the fully connected layer in the regression head. """
    cls_head_dropout: float
    """ Dropout probability for the classification head. """
    reg_head_dropout: float
    """ Dropout probability for the regression head. """


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


class ChessModelResNet(ChessModel):
    """
    A ResNet-like model for classifying chess board states.
    - Input: A tensor of shape (N, 21, 8, 8) where N is the batch size.
    - Output: A tensor of logits of shape (N, 1968).
    """

    PROFILE: dict[str, ChessModelResnetParams] = {
        "fast": ChessModelResnetParams(
            res_block_sizes=[32, 64],
            res_block_counts=[3, 3],
            cls_head_size=128,
            reg_head_size=64,
            cls_head_dropout=0.0,
            reg_head_dropout=0.0,
        ),
        "default": ChessModelResnetParams(
            res_block_sizes=[64, 128],
            res_block_counts=[2, 2],
            cls_head_size=256,
            reg_head_size=128,
            cls_head_dropout=0.2,
            reg_head_dropout=0.2,
        ),
        "slow": ChessModelResnetParams(
            res_block_sizes=[128, 256],
            res_block_counts=[3, 3],
            cls_head_size=256,
            reg_head_size=128,
            cls_head_dropout=0.3,
            reg_head_dropout=0.3,
        ),
        "optimized": ChessModelResnetParams(
            res_block_sizes=[64],
            res_block_counts=[1],
            cls_head_size=256,
            reg_head_size=64,
            cls_head_dropout=0.0,
            reg_head_dropout=0.3,
        ),
    }

    def __init__(self, input_shape: tuple[int, int, int], output_shape: tuple[int], params: ChessModelParams):
        super().__init__()

        # =============================================================================
        # Handle arguments
        # =============================================================================

        output_width = output_shape[0]
        input_channels, input_height, input_width = input_shape

        # Validate and convert parameters
        # - Sometime we might get a base class, so we need get the default of the base class as a placeholder
        assert isinstance(params, ChessModelParams), "params must be an instance of ChessModelParams"
        resnet_params: ChessModelResnetParams = params if isinstance(params, ChessModelResnetParams) else ChessModelResNet.PROFILE["fast"]

        # sanity checks
        assert len(resnet_params.res_block_sizes) == len(
            resnet_params.res_block_counts
        ), f"res_block_sizes and res_block_counts must have the same length, got {len(resnet_params.res_block_sizes)} and {len(resnet_params.res_block_counts)}"

        # =============================================================================
        # Build the residual blocks
        # =============================================================================

        res_block_sizes = resnet_params.res_block_sizes
        res_block_counts = resnet_params.res_block_counts
        res_layers: list[nn.Module] = []
        for layer_idx in range(len(res_block_sizes)):
            in_channels = input_channels if layer_idx == 0 else res_block_sizes[layer_idx - 1]
            out_channels = res_block_sizes[layer_idx]

            # Initial conv layer to change number of channels
            res_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
            res_layers.append(nn.BatchNorm2d(out_channels))
            res_layers.append(nn.ReLU())

            # Add the specified number of residual blocks
            for _ in range(res_block_counts[layer_idx]):
                res_layers.append(ChessModelResNet_ResidualBlock(out_channels))

        self.res_layers = nn.Sequential(
            *res_layers,
            torch.nn.Flatten(),
        )

        # =============================================================================
        # Build the classification head (move prediction))
        # =============================================================================

        flat_features = res_block_sizes[-1] * 8 * 8

        # classification head (move prediction)
        self.cls_head = nn.Sequential(
            # fully connected layers
            nn.Linear(flat_features, resnet_params.cls_head_size),
            nn.ReLU(),
            nn.Dropout(resnet_params.cls_head_dropout),
            # Final classifier layer
            nn.Linear(resnet_params.cls_head_size, output_width),
        )

        # =============================================================================
        # Build the regression head (eval prediction)
        # =============================================================================

        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(flat_features, resnet_params.reg_head_size),
            nn.BatchNorm1d(resnet_params.reg_head_size),
            torch.nn.ReLU(),
            nn.Dropout(resnet_params.reg_head_dropout),
            torch.nn.Linear(resnet_params.reg_head_size, 1),
        )

    def forward(self, x):
        # x: (batch, 21, 8, 8)
        features = x.to(torch.float32)

        # apply the residual towers
        features = self.res_layers(features)

        # apply the move head
        move_logits = self.cls_head(features)

        # apply the eval head
        eval_pred = self.reg_head(features)

        # return both predictions
        return move_logits, eval_pred


# =============================================================================
# AlphaZero exact model - NOT USED
# =============================================================================


@dataclasses.dataclass
class ChessModelAlphaZeroNetParams(ChessModelParams):
    """
    Parameters for the AlphaZeroNet model.
    """

    res_block_count: int


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


class AlphaZeroNet(ChessModel):

    PROFILE: dict[str, ChessModelAlphaZeroNetParams] = {
        "default": ChessModelAlphaZeroNetParams(
            res_block_count=10,
        ),
    }

    def __init__(self, input_shape: tuple[int, int, int], output_shape: tuple[int], params: ChessModelParams):
        super().__init__()

        # =============================================================================
        # Handle arguments
        # =============================================================================

        in_channels, board_size, _ = input_shape
        action_size = output_shape[0]

        # Validate and convert parameters
        # - Sometime we might get a base class, so we need get the default of the base class as a placeholder
        assert isinstance(params, ChessModelParams), "params must be an instance of ChessModelParams"
        alphaZeroNet_params: ChessModelAlphaZeroNetParams = params if isinstance(params, ChessModelAlphaZeroNetParams) else AlphaZeroNet.PROFILE["default"]

        # =============================================================================
        #
        # =============================================================================

        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.res_blocks = nn.ModuleList([AlphaZeroNet_ResidualBlock(256) for _ in range(alphaZeroNet_params.res_block_count)])

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
