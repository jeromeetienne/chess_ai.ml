# pip imports
import torch

# local imports
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    # TODO could be replaced by `TensorDataset` from torch.utils.data import TensorDataset
    def __init__(self, boards_tensor: torch.Tensor, moves_tensor: torch.Tensor):
        self.boards_tensor = boards_tensor
        self.moves_tensor = moves_tensor

    def __len__(self):
        return len(self.boards_tensor)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.boards_tensor[idx], self.moves_tensor[idx]

