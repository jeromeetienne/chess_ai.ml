# stdlib imports
import os

# pip imports
import torch
import chess
import chess.pgn
import chess.polyglot
from tqdm import tqdm

# local imports
from src.libs.chess_extra import ChessExtra
from src.libs.encoding import Encoding
from src.utils.uci2class_utils import Uci2ClassUtils
from src.utils.pgn_utils import PGNUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "../../data")
tensor_folder_path = os.path.join(data_folder_path, "pgn_tensors")


class DatasetUtils:
    """
    Utility class for handling chess datasets, including loading, saving, and converting between PGN files and tensor representations.

    NOTE: a "dataset" here refers to a pair of tensors: (boards_tensor, moves_tensor). 
    Boards and moves are typically generated from a PGN file.
    The evals_tensor is handled separately as it is generated separately.
    """

    class FILE_SUFFIX:
        BOARDS = "_boards_tensor.pt"
        MOVES = "_moves_tensor.pt"
        EVALS = "_evals_tensor.pt"

    ###############################################################################
    #   File path helpers
    #

    @staticmethod
    def boards_tensor_path(tensors_folder_path: str, basename_prefix: str) -> str:
        return f"{tensors_folder_path}/{basename_prefix}{DatasetUtils.FILE_SUFFIX.BOARDS}"
    
    @staticmethod
    def moves_tensor_path(tensors_folder_path: str, basename_prefix: str) -> str:
        return f"{tensors_folder_path}/{basename_prefix}{DatasetUtils.FILE_SUFFIX.MOVES}"

    @staticmethod
    def evals_tensor_path(tensors_folder_path: str, basename_prefix: str) -> str:
        return f"{tensors_folder_path}/{basename_prefix}{DatasetUtils.FILE_SUFFIX.EVALS}"


    ###############################################################################
    #   Load/Save dataset tensors
    #

    @staticmethod
    def save_boards_tensor(boards_tensor: torch.Tensor, tensors_folder_path: str, basename_prefix: str) -> None:
        # save boards tensor
        boards_path = DatasetUtils.boards_tensor_path(tensors_folder_path, basename_prefix)
        torch.save(boards_tensor, boards_path)  

    @staticmethod
    def load_boards_tensor(tensors_folder_path: str, basename_prefix: str) -> torch.Tensor:
        # load the boards tensor
        boards_path = DatasetUtils.boards_tensor_path(tensors_folder_path, basename_prefix)
        boards_tensor = torch.load(boards_path)
        return boards_tensor
    
    @staticmethod
    def save_moves_tensor(moves_tensor: torch.Tensor, tensors_folder_path: str, basename_prefix: str) -> None:
        # save moves tensor
        moves_path = DatasetUtils.moves_tensor_path(tensors_folder_path, basename_prefix)
        torch.save(moves_tensor, moves_path)

    @staticmethod
    def load_moves_tensor(tensors_folder_path: str, basename_prefix: str) -> torch.Tensor:
        # load the moves tensor
        moves_path = DatasetUtils.moves_tensor_path(tensors_folder_path, basename_prefix)
        moves_tensor = torch.load(moves_path)
        return moves_tensor

    @staticmethod
    def save_evals_tensor(evals_tensor: torch.Tensor, tensors_folder_path: str, basename_prefix: str) -> None:
        # save evals tensor
        evals_path = DatasetUtils.evals_tensor_path(tensors_folder_path, basename_prefix)
        torch.save(evals_tensor, evals_path)

    @staticmethod
    def load_evals_tensor(tensors_folder_path: str, basename_prefix: str) -> torch.Tensor:
        # load the evals tensor
        evals_path = DatasetUtils.evals_tensor_path(tensors_folder_path, basename_prefix)
        evals_tensor = torch.load(evals_path)
        return evals_tensor

    @staticmethod
    def load_dataset(tensors_folder_path: str, basename_prefix: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # load the tensors
        boards_tensor = DatasetUtils.load_boards_tensor(tensors_folder_path, basename_prefix)
        moves_tensor = DatasetUtils.load_moves_tensor(tensors_folder_path, basename_prefix)
        evals_tensor = DatasetUtils.load_evals_tensor(tensors_folder_path, basename_prefix)
        # ensure they have the same number of positions
        assert (
            boards_tensor.shape[0] == moves_tensor.shape[0]
        ), f"boards_tensor has {boards_tensor.shape[0]} positions but moves_tensor has {moves_tensor.shape[0]} positions. basename_prefix={basename_prefix}"
        # return the dataset
        return boards_tensor, moves_tensor, evals_tensor

    @staticmethod
    def load_datasets(tensors_folder_path: str, max_file_count: int = 15) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # gather all tensor file paths
        basenames = sorted(os.listdir(tensors_folder_path))
        boards_basenames = [basename for basename in basenames if basename.endswith(DatasetUtils.FILE_SUFFIX.BOARDS)]
        basename_prefixes = [basename[:-len(DatasetUtils.FILE_SUFFIX.BOARDS)] for basename in boards_basenames]

        # honor the max_file_count limit
        if max_file_count != 0:
            basename_prefixes = basename_prefixes[:max_file_count]

        # load all tensors
        boards_tensors = []
        moves_tensors = []
        evals_tensors = []
        for basename_prefix in basename_prefixes:
            _boards_tensor = DatasetUtils.load_boards_tensor(tensors_folder_path, basename_prefix)
            _moves_tensor = DatasetUtils.load_moves_tensor(tensors_folder_path, basename_prefix)
            _evals_tensor = DatasetUtils.load_evals_tensor(tensors_folder_path, basename_prefix)
            boards_tensors.append(_boards_tensor)
            moves_tensors.append(_moves_tensor)
            evals_tensors.append(_evals_tensor)

        # ensure they have the same number of positions
        boards_count = sum([len(boards_tensor) for boards_tensor in boards_tensors])
        moves_count = sum([len(moves_tensor) for moves_tensor in moves_tensors])
        evals_count = sum([len(evals_tensor) for evals_tensor in evals_tensors])
        assert boards_count == moves_count, f"boards_tensor has {boards_count} positions but moves_tensor has {moves_count} positions"
        assert boards_count == evals_count, f"boards_tensor has {boards_count} positions but evals_tensor has {evals_count} positions"

        # log the event
        print(f"Loading dataset from {len(basename_prefixes)} files, total {boards_count:,} positions")

        # concatenate all tensors
        boards_tensor = torch.cat(boards_tensors, dim=0)
        moves_tensor = torch.cat(moves_tensors, dim=0)
        evals_tensor = torch.cat(evals_tensors, dim=0)

        # return the dataset
        return boards_tensor, moves_tensor, evals_tensor

    ###############################################################################
    #   Convert PGN games to boards and moves
    #

    @staticmethod
    def games_to_boards_moves(
        games: list[chess.pgn.Game], polyglot_reader: chess.polyglot.MemoryMappedReader | None
    ) -> tuple[list[chess.Board], list[chess.Move]]:
        boards: list[chess.Board] = []
        moves: list[chess.Move] = []
        for game in games:
            board = game.board()
            for move in game.mainline_moves():
                # backup the board before playing the move
                board_before = board.copy(stack=False)

                # push the move to the board
                board.push(move)

                # skip if the position is in the opening book
                if polyglot_reader and ChessExtra.is_in_opening_book(board, polyglot_reader):
                    continue

                # append the board in pgn_boards
                boards.append(board_before)

                # append the move in pgn_moves
                move_copy = chess.Move(move.from_square, move.to_square, promotion=move.promotion)
                moves.append(move_copy)

        return boards, moves

    @staticmethod
    def boards_moves_to_tensor(
        boards: list[chess.Board], moves: list[chess.Move]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(boards) == len(moves), f"len(boards)={len(boards)} != len(moves)={len(moves)}"

        # create tensors to hold the boards and moves
        boards_tensor = torch.zeros((len(boards), *Encoding.get_input_shape()), dtype=Encoding.BOARD_DTYPE)
        moves_tensor = torch.zeros((len(boards),), dtype=Encoding.MOVE_DTYPE)

        # iterate through all positions and encode them
        for position_index, (board, move) in enumerate(zip(boards, moves)):
            # encode the current board position
            board_tensor = Encoding.board_to_tensor(board)
            boards_tensor[position_index] = board_tensor

            # encode the best move in UCI format
            moves_tensor[position_index] = Encoding.move_to_tensor(move.uci(), board.turn)

        # return the boards, moves and the mapping
        return boards_tensor, moves_tensor

    ###############################################################################
    #   Check dataset integrity
    #

    @staticmethod
    def check_tensor_vs_pgn(pgn_path: str, polyglot_reader: chess.polyglot.MemoryMappedReader | None, verbose: bool = False) -> int:
        """
        Check the integrity of the dataset by comparing a PGN file to its tensor representation.
        """
        pgn_basename = os.path.basename(pgn_path).replace(".pgn", "")

        print(f"Loading tensors for {pgn_basename}")
        boards_tensor, moves_tensor, evals_tensor = DatasetUtils.load_dataset(tensor_folder_path, pgn_basename)

        ###############################################################################
        #   convert the pgn games in boards and moves, skipping the opening book positions

        print(f"Checking {pgn_basename} - {str(len(boards_tensor)).rjust(5)} positions")

        # parse the pgn file
        pgn_games = PGNUtils.pgn_file_to_games(pgn_path)

        # split the games into boards and moves
        pgn_boards, pgn_moves = DatasetUtils.games_to_boards_moves(pgn_games, polyglot_reader)

        ###############################################################################
        #   Load the tensors for this pgn file
        #

        assert len(pgn_boards) == len(boards_tensor), f"len(pgn_boards)={len(pgn_boards)} != len(boards_tensor)={len(boards_tensor)}"
        assert len(pgn_moves) == len(moves_tensor), f"len(pgn_moves)={len(pgn_moves)} != len(moves_tensor)={len(moves_tensor)}"

        if verbose:
            print(f"Comparing {str(len(pgn_boards)).rjust(5)} positions between tensors and PGN")

        # convert boards_tensor and moves_tensor to chess.Board and chess.Move
        difference_count = 0
        for i in range(boards_tensor.shape[0]):
            pgn_board = pgn_boards[i]

            # encode the board
            tensor_board = Encoding.board_from_tensor(boards_tensor[i])

            # Check if the board positions are equal using FEN
            board_is_equal = pgn_board.board_fen() == tensor_board.board_fen()
            if not board_is_equal:
                difference_count += 1
                if verbose:
                    print(f"Boards are not equal at index {i}")
                    print(f"pgn_board.fen()   = {pgn_board.fen()}")
                    print(f"tensor_board.fen()= {tensor_board.fen()}")

            # encode the move
            pgn_move = pgn_moves[i]
            move_uci = Encoding.move_from_tensor(moves_tensor[i], tensor_board.turn)
            tensor_move = chess.Move.from_uci(move_uci)

            # Check if the moves are equal
            if pgn_move != tensor_move:
                difference_count += 1
                if verbose:
                    print(f"Moves are not equal at index {i}")
                    print(f"pgn_move   = {pgn_move}")
                    print(f"tensor_move= {tensor_move}")

            if verbose and i % 100 == 0:
                print(".", end="", flush=True)

        return difference_count

    ###############################################################################
    #   Dataset summary
    #

    @staticmethod
    def dataset_summary(boards_tensor: torch.Tensor, moves_tensor: torch.Tensor, evals_tensor: torch.Tensor) -> str:
        summary = f"""Dataset Summary:
- Total positions: {len(boards_tensor):,}
- Input: size {Encoding.get_input_shape()} (Channels, Height, Width), type {Encoding.BOARD_DTYPE}
- Output shape: size {moves_tensor.shape} (Scalar class index), type {Encoding.MOVE_DTYPE}
"""
        return summary



    @staticmethod
    def normalize_evals_tensor(evals_tensor: torch.Tensor) -> torch.Tensor:
        """ Normalize evals to be between -1 and 1 using tanh function. """
        return torch.tanh(evals_tensor / 10.0)

    @staticmethod
    # Inverse transform function for later use
    def denormalize_evals_tensor(normalized_evals_tensor: torch.Tensor) -> torch.Tensor:
        """ Inverse of normalize_evals_tensor with atanh function. """
        return torch.atanh(normalized_evals_tensor) * 10.0

    @staticmethod
    def tensor_histogram_ascii(tensor: torch.Tensor, bins: int = 10, width: int = 50) -> str:
        hist, bin_edges = torch.histogram(tensor, bins=bins)
        hist = hist.cpu().numpy()
        bin_edges = bin_edges.cpu().numpy()

        max_count = hist.max()
        scale = width / max_count if max_count > 0 else 1

        histogram_lines = []
        for count, edge_start, edge_end in zip(hist, bin_edges[:-1], bin_edges[1:]):
            bar = "*" * int(count * scale)
            histogram_lines.append(f"{edge_start:6.2f} - {edge_end:6.2f} | {bar} ({count})")

        return "\n".join(histogram_lines)
    


###############################################################################
#   Example usage
#
if __name__ == "__main__":
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    # board.push(chess.Move.from_uci("a7a5"))
    # move = chess.Move.from_uci("c2c4")

    move = chess.Move.from_uci("h7h5")

    print(f"Current board: {'white' if board.turn == chess.WHITE else 'black'} move {move.uci()}\n{board}")

    board_tensor = Encoding.board_to_tensor(board)
    move_tensor = Encoding.move_to_tensor(move.uci(), board.turn)

    print(f"move_tensor: {move_tensor}")

    reconstructed_board = Encoding.board_from_tensor(board_tensor)
    reconstructed_move_uci = Encoding.move_from_tensor(move_tensor, board.turn)
    reconstructed_move = chess.Move.from_uci(reconstructed_move_uci)

    print(f"Reconstructed board: {'white' if reconstructed_board.turn == chess.WHITE else 'black'} move {reconstructed_move.uci()}")
    print(reconstructed_board)
