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
    @staticmethod
    def dataset_summary(boards_tensor: torch.Tensor, moves_tensor: torch.Tensor) -> str:
            summary = f"""Dataset Summary:
- Total positions: {len(boards_tensor):,}
- Input: size {Encoding.INPUT_SHAPE} (Channels, Height, Width), type {Encoding.BOARD_DTYPE}
- Output shape: size {moves_tensor.shape} (Scalar class index), type {Encoding.MOVE_DTYPE}
"""
            return summary

    @staticmethod
    def save_dataset(boards_tensor: torch.Tensor, moves_tensor: torch.Tensor, folder_path: str):
        # os.makedirs(folder_path, exist_ok=True)
        boards_path = f"{folder_path}/dataset_boards.pt"
        moves_path = f"{folder_path}/dataset_moves.pt"
        torch.save(boards_tensor, boards_path)
        torch.save(moves_tensor, moves_path)

    @staticmethod
    def load_dataset_tensor(tensors_folder_path: str, basename_prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        # load the tensors
        boards_path = f"{tensors_folder_path}/{basename_prefix}_boards_tensor.pt"
        boards_tensor = torch.load(boards_path)
        # load the moves tensor
        moves_path = f"{tensors_folder_path}/{basename_prefix}_moves_tensor.pt"
        moves_tensor = torch.load(moves_path)
        # ensure they have the same number of positions
        assert boards_tensor.shape[0] == moves_tensor.shape[0], f"boards_tensor has {boards_tensor.shape[0]} positions but moves_tensor has {moves_tensor.shape[0]} positions. basename_prefix={basename_prefix}"
        # return the dataset
        return boards_tensor, moves_tensor

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
    def split_game_to_board_move(games: list[chess.pgn.Game], polyglot_reader: chess.polyglot.MemoryMappedReader | None) -> tuple[list[chess.Board], list[chess.Move]]:
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
        
        # split the games into boards and moves
        boards, moves = DatasetUtils.split_game_to_board_move(games, polyglot_reader)


        # create tensors to hold the boards and moves
        boards_tensor = torch.zeros((len(boards), *Encoding.INPUT_SHAPE), dtype=Encoding.BOARD_DTYPE)
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

    @staticmethod
    def check_tensor_from_pgn(pgn_path: str, polyglot_reader: chess.polyglot.MemoryMappedReader | None, verbose: bool = False) -> int:
        pgn_basename = os.path.basename(pgn_path).replace(".pgn", "")

        print(f'Loading tensors for {pgn_basename}')
        boards_tensor, moves_tensor = DatasetUtils.load_dataset_tensor(tensor_folder_path, pgn_basename)

        ###############################################################################
        #   convert the pgn games in boards and moves, skipping the opening book positions

        print(f'Parsing {pgn_basename} - {str(len(boards_tensor)).rjust(5)} positions')

        # parse the pgn file
        pgn_games = PGNUtils.parse_pgn_file(pgn_path)

        # split the games into boards and moves
        pgn_boards, pgn_moves = DatasetUtils.split_game_to_board_move(pgn_games, polyglot_reader)

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

            if verbose and i%100 == 0:
                print(".", end="", flush=True)


        return difference_count

if __name__ == "__main__":
    ###############################################################################
    #   Example usage
    #
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    # board.push(chess.Move.from_uci("a7a5"))
    # move = chess.Move.from_uci("c2c4")

    move = chess.Move.from_uci("h7h5")

    print(f"Current board: {'white' if board.turn == chess.WHITE else 'black'} move {move.uci()}\n{board}")

    board_tensor = Encoding.board_to_tensor(board)
    move_tensor = Encoding.move_to_tensor(move.uci(), board.turn)

    print(f'move_tensor: {move_tensor}')

    reconstructed_board = Encoding.board_from_tensor(board_tensor)
    reconstructed_move_uci = Encoding.move_from_tensor(move_tensor, board.turn)
    reconstructed_move = chess.Move.from_uci(reconstructed_move_uci)

    print(f"Reconstructed board: {'white' if reconstructed_board.turn == chess.WHITE else 'black'} move {reconstructed_move.uci()}")
    print(reconstructed_board)