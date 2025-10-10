# pip imports
import numpy as np
import torch
import chess
import chess.pgn
import chess.polyglot
from tqdm import tqdm

# local imports
from src.libs.chess_extra import ChessExtra
from src.utils.uci2class_utils import Uci2ClassUtils
from src.libs.encoding import Encoding


class MoveEncodingUci2Class:

    @staticmethod
    def move_to_tensor(move_uci: str, color: chess.Color) -> torch.Tensor:
        uci2class = Uci2ClassUtils.get_uci2class(color)
        class_index = uci2class[move_uci]
        move_tensor = torch.tensor(class_index, dtype=Encoding.MOVE_DTYPE)
        return move_tensor

    @staticmethod
    def move_from_tensor(moves_tensor: torch.Tensor, color: chess.Color) -> str:
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
        return move_uci


# =============================================================================
# Usage example
# =============================================================================
if __name__ == "__main__":
    move = chess.Move.from_uci("a2a3")
    turn = chess.WHITE

    move_tensor = MoveEncodingUci2Class.move_to_tensor(move.uci(), turn)
    reconstructed_move_uci = MoveEncodingUci2Class.move_from_tensor(move_tensor, turn)
    reconstructed_move = chess.Move.from_uci(reconstructed_move_uci)
    assert move == reconstructed_move

    print(f"Original move: {move}, UCI: {move.uci()}")
    print(f"Move tensor: {move_tensor}")
    print(f"Reconstructed move from tensor: {reconstructed_move}, UCI: {reconstructed_move.uci()}")
    ###############################################################################
