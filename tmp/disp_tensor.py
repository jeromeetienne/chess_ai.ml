# stdlib imports
import os
import sys

# pip imports
import chess
import torch
import chess.polyglot

# local imports
from src.utils.dataset_utils import DatasetUtils
from src.utils.pgn_utils import PGNUtils
from src.libs.chess_extra import ChessExtra
from src.encoding.board_encoding import BoardEncoding


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "..", "data")
tensor_folder_path = os.path.join(data_folder_path, "pgn_tensors")


if __name__ == "__main__":

    # Load polyglot opening book
    polyglot_path = os.path.join(data_folder_path, "polyglot/lichess_pro_books/lpb-allbook.bin")
    polyglot_reader = chess.polyglot.open_reader(polyglot_path)

    # Get the first pgn file
    pgn_paths = PGNUtils.get_pgn_paths()
    pgn_path = pgn_paths[0]

    # parse the pgn file
    pgn_games = PGNUtils.pgn_file_to_games(pgn_path)

    # Go thru all the moves of the game, and store the board and move if the position is not in the opening book
    pgn_boards, pgn_moves, pgn_moves_index = DatasetUtils.games_to_boards_moves(pgn_games, polyglot_reader)

    pgn_basename = os.path.basename(pgn_path).replace(".pgn", "")
    boards_tensor, moves_tensor, evals_tensor, moves_index = DatasetUtils.load_dataset(tensor_folder_path, pgn_basename)

    ###############################################################################
    #   Display some info
    #
    print(f"Boards tensor shape: {boards_tensor.shape}")
    print(f"Moves tensor shape: {moves_tensor.shape}")

    for board_index in range(10):
        pgn_board = pgn_boards[board_index]
        pgn_move = pgn_moves[board_index]
        print(f"pgn board:")
        print(pgn_board)
        print(f"pgn move: {pgn_move.uci()}")

        board_tensor = boards_tensor[board_index]

        move_tensor = moves_tensor[board_index]
        print(f"First board tensor: {'white' if pgn_board.turn else 'black'}")
        # print(board_tensor)

        board_numpy = board_tensor.numpy()
        feature_plane = board_numpy[BoardEncoding.PLANE.ACTIVE_PAWN]
        print(f"Feature plane for active pawn:")
        print(feature_plane)

        print("First move tensor:")
        print(move_tensor)
