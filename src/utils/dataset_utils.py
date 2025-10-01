# stdlib imports
import os

# pip imports
import torch
import chess
import chess.pgn
import chess.polyglot
from tqdm import tqdm

# local imports
from ..libs.chess_extra import ChessExtra
from ..libs.encoding import Encoding
from .uci2class_utils import Uci2ClassUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "../../data")


class DatasetUtils:
    @staticmethod
    def dataset_summary(boards_tensor: torch.Tensor, moves_tensor: torch.Tensor) -> str:
            summary = f"""Dataset Summary:
- Total positions: {len(boards_tensor):,}
- Input shape: {boards_tensor.shape[1:]} (Channels, Height, Width)
- Output shape: {moves_tensor.shape[1:]} (Scalar class index)
"""
            # FIXME the output share is super crappy
            return summary

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
    def games_to_tensor(
        games: list[chess.pgn.Game],
        polyglot_reader: chess.polyglot.MemoryMappedReader | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a list of chess games into boards tensor. Skip positions that are in the opening book.

        Args:
            games (list[chess.pgn.Game]): A list of chess games in PGN format.
            polyglot_reader (chess.polyglot.MemoryMappedReader | None): A polyglot reader for opening book moves.
        Returns:
            tuple:
                - torch.Tensor: **boards_tensor** A tensor containing encoded board positions with shape (num_positions, *INPUT_SHAPE).
                - torch.Tensor: **moves_tensor** A tensor of class indices representing the best move for each position.

        The function iterates through all positions in the provided games, encodes each board state into a tensor,
        and collects the best move in UCI format. It then creates a mapping from unique UCI moves to class indices,
        encodes the moves as class indices, and returns the board tensors, move tensors, and the mapping.
        """
        ###############################################################################
        #   Count total positions
        #
        position_max_count = 0
        for game in games:
            game_move_count = len(list(game.mainline_moves()))
            position_max_count += game_move_count

        ###############################################################################
        #   Encode boards and moves
        #

        uci2class_white = Uci2ClassUtils.get_uci2class(chess.WHITE)
        uci2class_black = Uci2ClassUtils.get_uci2class(chess.BLACK)
        
        boards_tensor = torch.zeros((position_max_count, *Encoding.INPUT_SHAPE), dtype=Encoding.BOARD_DTYPE)
        moves_tensor = torch.zeros((position_max_count,), dtype=Encoding.MOVE_DTYPE)
        position_index = 0
        for game in games:
            board = game.board()
            for move in game.mainline_moves():
                # Play this move on the board to get to the next position
                board.push(move)

                # skip positions that are in the opening book
                if polyglot_reader is not None and ChessExtra.is_in_opening_book(board, polyglot_reader):
                    continue

                # encode the current board position
                board_tensor = Encoding.board_to_tensor(board)
                boards_tensor[position_index] = board_tensor

                # encode the best move in UCI format
                moves_tensor[position_index] = uci2class_white[move.uci()] if board.turn == chess.WHITE else uci2class_black[move.uci()]
 
                # Update the board index
                position_index += 1

        # truncate boards_tensor to the actual number of positions encoded
        boards_tensor = boards_tensor[:position_index, :, :, :]

        # return the boards, moves and the mapping
        return boards_tensor, moves_tensor

