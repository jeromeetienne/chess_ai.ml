# stdlib imports
import os

# pip imports
import torch

# local imports
from ..libs.chess_model import ChessModel, ChessModelParams, ChessModelConv2d, ChessModelFullConv, ChessModelResNet, AlphaZeroNet
from ..encoding.board_encoding import BoardEncoding
from ..encoding.move_encoding import MoveEncoding

__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "../../data")


class ModelUtils:
    class MODEL_NAME:
        CHESS_MODEL_CONV2D = "ChessModelConv2d"
        CHESS_MODEL_FULL_CONV = "ChessModelFullConv"
        CHESS_MODEL_RESNET = "ChessModelResNet"
        ALPHA_ZERO_NET = "AlphaZeroNet"

    @staticmethod
    def get_supported_models() -> list[str]:
        # get supported models from MODEL_NAME class
        supported_models = [value for name, value in vars(ModelUtils.MODEL_NAME).items() if not name.startswith("__") and not callable(value)]
        return supported_models

    @staticmethod
    def create_model(model_name: str, model_params: ChessModelParams) -> ChessModel:
        input_shape, output_shape = BoardEncoding.get_input_shape(), MoveEncoding.get_shape_tensor_output()
        # Create the model
        if model_name == ModelUtils.MODEL_NAME.CHESS_MODEL_CONV2D:
            model = ChessModelConv2d(input_shape=input_shape, output_shape=output_shape, params=model_params)
        elif model_name == ModelUtils.MODEL_NAME.CHESS_MODEL_FULL_CONV:
            model = ChessModelFullConv(input_shape=input_shape, output_shape=output_shape, params=model_params)
        elif model_name == ModelUtils.MODEL_NAME.CHESS_MODEL_RESNET:
            model = ChessModelResNet(input_shape=input_shape, output_shape=output_shape, params=model_params)
        elif model_name == ModelUtils.MODEL_NAME.ALPHA_ZERO_NET:
            model = AlphaZeroNet(input_shape=input_shape, output_shape=output_shape, params=model_params)
        else:
            assert False, f"Unknown model name: {model_name}"

        return model

    @staticmethod
    def model_path(folder_path: str) -> str:
        model_path = f"{folder_path}/model.pth"
        return model_path

    # =============================================================================
    # load/save model
    # =============================================================================

    @staticmethod
    def load_model(model_name: str, folder_path: str, model_params: ChessModelParams) -> torch.nn.Module:
        model = ModelUtils.create_model(model_name, model_params=model_params)
        ModelUtils.load_weights(model, folder_path)
        return model

    @staticmethod
    def load_weights(model: torch.nn.Module, folder_path: str) -> None:
        # Load the model
        model_path = ModelUtils.model_path(folder_path)
        model.load_state_dict(torch.load(model_path))

    @staticmethod
    def save_model(model: torch.nn.Module, folder_path: str):
        # Save the model
        model_path = ModelUtils.model_path(folder_path)
        torch.save(model.state_dict(), model_path)

    @staticmethod
    def guess_model_name_profile(folder_path: str) -> tuple[str | None, str | None]:
        """Try to guess the model name and profile by attempting to load the model with each supported model and profile.

        - it just try them all until one works

        Args:
            folder_path (str): The folder path where the model is saved.

        Returns:
            tuple[str | None, str | None]: The guessed model name and profile, or (None, None) if not found.
        """
        model_names = ModelUtils.get_supported_models()
        for model_name in model_names:
            # =============================================================================
            # Get the model class
            # =============================================================================
            model_class = None
            if model_name == ModelUtils.MODEL_NAME.CHESS_MODEL_CONV2D:
                model_class = ChessModelConv2d
            elif model_name == ModelUtils.MODEL_NAME.CHESS_MODEL_RESNET:
                model_class = ChessModelResNet
            elif model_name == ModelUtils.MODEL_NAME.ALPHA_ZERO_NET:
                model_class = AlphaZeroNet
            else:
                assert False, f"Unknown model name: {model_name}"

            profile_names = model_class.PROFILE.keys()
            for profile_name in profile_names:
                model_params: ChessModelParams = model_class.PROFILE[profile_name]
                model = ModelUtils.create_model(model_name, model_params=model_params)
                try:
                    ModelUtils.load_model(model_name, folder_path, model_params=model_params)

                    # return the first one that works
                    return model_name, profile_name
                except Exception:
                    continue

        return None, None
