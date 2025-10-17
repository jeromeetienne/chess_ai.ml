# pip imports
import numpy as np
import torch
import chess

# local imports
from src.utils.uci2class_utils import Uci2ClassUtils


class MoveEncodingUci2Class:
    MOVE_DTYPE = torch.int32  # class index as long

    # create a static property accesor for .OUTPUT_SHAPE
    @staticmethod
    def get_output_shape() -> tuple[int]:
        uci2class_white = Uci2ClassUtils.get_uci2class(chess.WHITE)
        num_classes = len(uci2class_white)
        return (num_classes,)

    # =============================================================================
    # move_to_tensor/move_from_tensor
    # =============================================================================
    @staticmethod
    def encode_move_tensor(move: chess.Move, color: chess.Color) -> torch.Tensor:
        uci2class = Uci2ClassUtils.get_uci2class(color)
        class_index = uci2class[move.uci()]
        move_tensor = torch.tensor(class_index, dtype=MoveEncodingUci2Class.MOVE_DTYPE)
        return move_tensor

    @staticmethod
    def decode_move_tensor(moves_tensor: torch.Tensor, color: chess.Color) -> chess.Move:
        """
        Converts a scalar move tensor representing a class index into its corresponding UCI move string.

        Arguments:
            moves_tensor (torch.Tensor): A scalar tensor containing the class index of the move.
            color (chess.Color): The color of the player making the move (chess.WHITE or chess.BLACK).
        Returns:
            str: The UCI string representation of the move.
        """
        class2uci = Uci2ClassUtils.get_class2uci(color)
        class_index = int(moves_tensor.item())
        move_uci = class2uci[class_index]
        move = chess.Move.from_uci(move_uci)
        return move


# =============================================================================
# Usage example
# =============================================================================
if __name__ == "__main__":
    move = chess.Move.from_uci("a2a3")
    turn = chess.WHITE

    move_tensor = MoveEncodingUci2Class.encode_move_tensor(move, turn)
    reconstructed_move = MoveEncodingUci2Class.decode_move_tensor(move_tensor, turn)
    assert move == reconstructed_move

    print(f"Original move: {move}, UCI: {move.uci()}")
    print(f"Move tensor: {move_tensor}")
    print(f"Reconstructed move from tensor: {reconstructed_move}, UCI: {reconstructed_move.uci()}")
    ###############################################################################
