import json
from typing_extensions import deprecated
from .chess_model import ChessModel
import torch
import pickle
import os
import chess


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "../../data")


class IOUtils:

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

    @staticmethod
    def save_dataset(boards_tensor: torch.Tensor, moves_tensor: torch.Tensor, folder_path: str):
        # os.makedirs(folder_path, exist_ok=True)
        boards_path = f"{folder_path}/dataset_boards.pt"
        moves_path = f"{folder_path}/dataset_moves.pt"
        torch.save(boards_tensor, boards_path)
        torch.save(moves_tensor, moves_path)

    @staticmethod
    def load_dataset(folder_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        # setup paths
        boards_path = f"{folder_path}/dataset_boards.pt"
        moves_path = f"{folder_path}/dataset_moves.pt"
        # load files
        boards_tensor = torch.load(boards_path)
        moves_tensor = torch.load(moves_path)

        # return the dataset
        return boards_tensor, moves_tensor

    @staticmethod
    def has_dataset(folder_path: str) -> bool:
        boards_path = f"{folder_path}/dataset_boards.pt"
        moves_path = f"{folder_path}/dataset_moves.pt"
        dataset_exists = os.path.exists(boards_path) and os.path.exists(moves_path)
        return dataset_exists

    @staticmethod
    @deprecated("Use uci2class_load instead")
    def load_uci_to_classindex(folder_path: str) -> dict[str, int]:
        """
        Load the uci_to_classindex mapping from a pickle file.
        Useful for inference when playing a game. when we dont want to load the entire dataset, because its large and slow
        Arguments:
            folder_path (str): Path to the folder containing the uci_to_classindex.pickle file.
        """
        assert False, "This method is deprecated. Use uci2class_load instead."
        uci_to_classindex_path = f"{folder_path}/uci_to_classindex.pickle"
        with open(uci_to_classindex_path, "rb") as file:
            uci_to_classindex = pickle.load(file)
        return uci_to_classindex

    @staticmethod
    @deprecated("Use uci2class_inverse_mapping instead")
    def classindex_to_uci_inverse_mapping(uci_to_classindex: dict[str, int]) -> dict[int, str]:
        assert False, "This method is deprecated. Use uci2class_inverse_mapping instead."
        classindex_to_uci = {v: k for k, v in uci_to_classindex.items()}
        return classindex_to_uci

    @staticmethod
    def uci2class_load(chess_color: chess.Color = chess.WHITE) -> dict[str, int]:
        color_name = "white" if chess_color == chess.WHITE else "black"
        file_path = os.path.join(data_folder_path, f"uci2class_arr_{color_name}.json")
        with open(file_path, "r") as file_reader:
            uci2class_arr: list[str] = json.load(file_reader)

        # Build the mapping
        uci2class = {move_uci: index for index, move_uci in enumerate(uci2class_arr)}

        return uci2class

    @staticmethod
    def uci2class_inverse_mapping(chess_color: chess.Color = chess.WHITE) -> dict[int, str]:
        uci2class = IOUtils.uci2class_load(chess_color)
        class_to_uci = {value: key for key, value in uci2class.items()}
        return class_to_uci
