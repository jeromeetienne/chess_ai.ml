# stdlib imports
import os

# pip imports
import torch
import chess.pgn

# local imports
from .chess_extra import ChessExtra
from .pgn_utils import PGNUtils
from .encoding import Encoding

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))


class Utils:
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
    def dataset_summary(boards_tensor: torch.Tensor, moves_tensor: torch.Tensor, uci_to_classindex: dict) -> str:
            summary = f"""Dataset Summary:
- Total positions: {len(boards_tensor):,}
- Input shape: {boards_tensor.shape[1:]} (Channels, Height, Width)
- Output shape: {moves_tensor.shape[1:]} (Scalar class index)
- Number of unique moves (classes): {len(uci_to_classindex):,}
- Sample move index (first position): {moves_tensor[0].item()}
- Sample move UCI (first position): {list(uci_to_classindex.keys())[list(uci_to_classindex.values()).index(moves_tensor[0].item())]}
"""
            return summary

