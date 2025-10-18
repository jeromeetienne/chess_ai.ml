# stdlib imports
import os

# pip imports
import torch

# local imports
from ..libs.chess_model import ChessModelConv2d, ChessModelResNet, AlphaZeroNet
from ..encoding.board_encoding import BoardEncoding
from ..encoding.move_encoding import MoveEncoding

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
        input_shape, output_shape = BoardEncoding.get_input_shape(), MoveEncoding.get_shape_tensor_output()
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
    def model_path(folder_path: str) -> str:
        model_path = f"{folder_path}/model.pth"
        return model_path

    @staticmethod
    def load_weights(model: torch.nn.Module, folder_path: str) -> None:
        # Load the model
        model_path = ModelUtils.model_path(folder_path)
        model.load_state_dict(torch.load(model_path))

    # =============================================================================
    # load/save model
    # =============================================================================

    @staticmethod
    def load_model(model_name: str, folder_path: str) -> torch.nn.Module:
        model = ModelUtils.create_model(model_name)
        ModelUtils.load_weights(model, folder_path)
        return model

    @staticmethod
    def save_model(model: torch.nn.Module, folder_path: str):
        # Save the model
        state_dict_path = f"{folder_path}/model.pth"
        torch.save(model.state_dict(), state_dict_path)
