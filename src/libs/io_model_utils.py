# stdlib imports
import os

# pip imports
import torch

# local imports
from .chess_model import ChessModel


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "../../data")


class IoModel:

    @staticmethod
    def save_model(model: ChessModel, folder_path: str):
        # Save the model
        state_dict_path = f"{folder_path}/model.pth"
        torch.save(model.state_dict(), state_dict_path)

    @staticmethod
    def load_model(folder_path: str, input_shape: tuple[int, int, int], output_shape: tuple[int]) -> ChessModel:
        # Load the model
        model = ChessModel(input_shape=input_shape, output_shape=output_shape)
        model_path = f"{folder_path}/model.pth"
        model.load_state_dict(torch.load(model_path))

        return model
