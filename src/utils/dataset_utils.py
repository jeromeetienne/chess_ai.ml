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
            uci2class_white = Uci2ClassUtils.get_uci2class(chess.WHITE)  # ensure the mappings are loaded
            summary = f"""Dataset Summary:
- Total positions: {len(boards_tensor):,}
- Input shape: {Encoding.BOARD_DTYPE} (Channels, Height, Width)
- Output shape: {Encoding.MOVE_DTYPE} (Scalar class index)
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
    def load_dataset(tensors_folder_path: str, max_file_count: int = 15) -> tuple[torch.Tensor, torch.Tensor]:
        # gather all tensor file paths
        basenames = os.listdir(tensors_folder_path)
        basenames.sort()
        boards_file_paths = [os.path.join(tensors_folder_path, basename) for basename in basenames if basename.endswith("_boards_tensor.pt")]
        moves_file_paths = [os.path.join(tensors_folder_path, basename) for basename in basenames if basename.endswith("_moves_tensor.pt")]

        # honor the max_file_count limit
        if max_file_count != 0:
            boards_file_paths = boards_file_paths[:max_file_count]
            moves_file_paths = moves_file_paths[:max_file_count]

        # load all tensors
        boards_tensors = []
        moves_tensors = []
        for boards_file_path, moves_file_path in zip(boards_file_paths, moves_file_paths):
            _boards_tensor = torch.load(boards_file_path)
            _moves_tensor = torch.load(moves_file_path)
            boards_tensors.append(_boards_tensor)
            moves_tensors.append(_moves_tensor)
            
        # Count total positions
        position_max_count = sum([boards_tensor.shape[0] for boards_tensor in boards_tensors])
        move_max_count = sum([moves_tensor.shape[0] for moves_tensor in moves_tensors])
        assert position_max_count == move_max_count, f"boards_tensor has {position_max_count} positions but moves_tensor has {move_max_count} positions"
        print(f"Loading dataset from {len(boards_file_paths)} files, total {position_max_count:,} positions")

        # concatenate all tensors
        boards_tensor = torch.cat(boards_tensors, dim=0)
        moves_tensor = torch.cat(moves_tensors, dim=0)

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
        moves_tensor = moves_tensor[:position_index]

        # return the boards, moves and the mapping
        return boards_tensor, moves_tensor

