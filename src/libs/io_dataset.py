# stdlib imports
import os

# pip imports
import torch


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "../../data")


class IoDataset:

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
