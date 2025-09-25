# pip imports
import torch

# local imports
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, boards_tensor: torch.Tensor, moves_tensor: torch.Tensor):
        self.boards_tensor = boards_tensor
        self.moves_tensor = moves_tensor

    def __len__(self):
        return len(self.boards_tensor)

    def __getitem__(self, idx):
        return self.boards_tensor[idx], self.moves_tensor[idx]

