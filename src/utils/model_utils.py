# stdlib imports
import os

# pip imports
import torch

from src.libs.encoding import Encoding

# local imports
from ..libs.chess_model import ChessModelConv2d, ChessModelResNet, AlphaZeroNet


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "../../data")


class ModelUtils:
    class MODEL_NAME:
        CHESS_MODEL_CONV2D = "ChessModelConv2d"
        CHESS_MODEL_RESNET = "ChessModelResNet"
        ALPHA_ZERO_NET = "AlphaZeroNet"

    @staticmethod
    def get_supported_models() -> list[str]:
        # get supported models from MODEL_NAME class
        supported_models = [value for name, value in vars(ModelUtils.MODEL_NAME).items() if not name.startswith("__") and not callable(value)]
        return supported_models

    @staticmethod
    def create_model(model_name: str) -> torch.nn.Module:
        input_shape, output_shape = Encoding.get_input_shape(), Encoding.get_output_shape()
        # Create the model
        if model_name == ModelUtils.MODEL_NAME.CHESS_MODEL_CONV2D:
            model = ChessModelConv2d(input_shape=input_shape, output_shape=output_shape)
        elif model_name == ModelUtils.MODEL_NAME.CHESS_MODEL_RESNET:
            model = ChessModelResNet(input_shape=input_shape, output_shape=output_shape)
        elif model_name == ModelUtils.MODEL_NAME.ALPHA_ZERO_NET:
            model = AlphaZeroNet(input_shape=input_shape, output_shape=output_shape)
        else:
            assert False, f"Unknown model name: {model_name}"

        return model

    @staticmethod
    def model_summary(model: torch.nn.Module) -> str:
        """
        Prints a basic summary of the model including parameter count.
        """
        total_params = 0
        trainable_params = 0

        output = []
        output.append(f"\nModel Architecture: {model.__class__.__name__}")
        output.append("-" * 60)
        output.append(f"{'Layer Name':<30} {'Param Count':>15} {'Trainable':>10}")
        output.append("=" * 60)

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            param = parameter.numel()
            total_params += param

            if parameter.requires_grad:
                trainable_params += param

            output.append(f"{name:<30} {param:>15,} {'Yes' if parameter.requires_grad else 'No':>10}")

        output.append("=" * 60)
        output.append(f"Total Parameters: {total_params:,}")
        output.append(f"Trainable Parameters: {trainable_params:,}")
        output.append(f"Non-trainable Parameters: {total_params - trainable_params:,}")
        output.append("-" * 60)

        return "\n".join(output)

    @staticmethod
    def save_model(model: torch.nn.Module, folder_path: str):
        # Save the model
        state_dict_path = f"{folder_path}/model.pth"
        torch.save(model.state_dict(), state_dict_path)

    @staticmethod
    def load_weights(model: torch.nn.Module, folder_path: str) -> None:
        # Load the model
        model_path = f"{folder_path}/model.pth"
        model.load_state_dict(torch.load(model_path))
