from .model import ChessModel
import torch
import pickle
import os

class IOUtils:


    @staticmethod
    def save_model(model: ChessModel, folder_path: str):
        # Save the model
        state_dict_path = f"{folder_path}/model.pth"
        torch.save(model.state_dict(), state_dict_path)

    @staticmethod
    def load_model(folder_path: str, num_classes: int) -> ChessModel:
        # Load the model
        model = ChessModel(num_classes=num_classes)
        model_path = f"{folder_path}/model.pth"
        model.load_state_dict(torch.load(model_path))

        return model

    @staticmethod
    def save_dataset(boards_tensor: torch.Tensor, moves_tensor: torch.Tensor, uci_to_classindex: dict[str, int], folder_path: str):
        # os.makedirs(folder_path, exist_ok=True)
        boards_path = f"{folder_path}/dataset_boards.pt"
        moves_path = f"{folder_path}/dataset_moves.pt"
        uci_to_classindex_path = f"{folder_path}/uci_to_classindex.pickle"
        torch.save(boards_tensor, boards_path)
        torch.save(moves_tensor, moves_path)
        with open(uci_to_classindex_path, "wb") as file:
            pickle.dump(uci_to_classindex, file)

    @staticmethod
    def load_dataset(folder_path: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        boards_path = f"{folder_path}/dataset_boards.pt"
        moves_path = f"{folder_path}/dataset_moves.pt"
        uci_to_classindex_path = f"{folder_path}/uci_to_classindex.pickle"
        boards_tensor = torch.load(boards_path)
        moves_tensor = torch.load(moves_path)
        with open(uci_to_classindex_path, "rb") as file:
            uci_to_classindex = pickle.load(file)
        return boards_tensor, moves_tensor, uci_to_classindex
    
    @staticmethod
    def has_dataset(folder_path: str) -> bool:
        boards_path = f"{folder_path}/dataset_boards.pt"
        moves_path = f"{folder_path}/dataset_moves.pt"
        uci_to_classindex_path = f"{folder_path}/uci_to_classindex.pickle"
        dataset_exists = os.path.exists(boards_path) and os.path.exists(moves_path) and os.path.exists(uci_to_classindex_path)
        return dataset_exists
            