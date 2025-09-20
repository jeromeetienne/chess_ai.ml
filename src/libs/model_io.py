from .model import ChessModel
import torch
import pickle

class ModelIO:
    @staticmethod
    def save_model(model: ChessModel, uci_to_classindex: dict[str, int], folder_path: str):
        # Save the model
        state_dict_path = f"{folder_path}/model.pth"
        torch.save(model.state_dict(), state_dict_path)

        # save uci_to_classindex mapping
        uci_to_classindex_path = f"{folder_path}/uci_to_classindex.pickle"
        with open(uci_to_classindex_path, "wb") as pgn_file_path:
            pickle.dump(uci_to_classindex, pgn_file_path)

    @staticmethod
    def load_model(folder_path: str) -> tuple[ChessModel, dict[str, int]]:
        # Load the uci_to_classindex mapping
        uci_to_classindex_path = f"{folder_path}/uci_to_classindex.pickle"
        with open(uci_to_classindex_path, "rb") as file:
            uci_to_classindex = pickle.load(file)

        # Load the model
        model = ChessModel(num_classes=len(uci_to_classindex))
        model_path = f"{folder_path}/model.pth"
        model.load_state_dict(torch.load(model_path))

        return model, uci_to_classindex
