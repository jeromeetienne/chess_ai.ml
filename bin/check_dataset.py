# !/usr/bin/env python3

# stdlib imports
import os

# pip imports
import chess.polyglot

# local imports
from src.libs.chess_extra import ChessExtra
from src.libs.encoding import Encoding
from src.utils.pgn_utils import PGNUtils
from src.utils.dataset_utils import DatasetUtils


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "..", "data")
pgn_folder_path = os.path.join(data_folder_path, "pgn")
tensor_folder_path = os.path.join(data_folder_path, "pgn_tensors")

if __name__ == "__main__":

    pgn_paths = PGNUtils.get_pgn_paths()

    pgn_index = 0
    pgn_path = pgn_paths[pgn_index]

    # Load polyglot opening book
    polyglot_path = os.path.join(data_folder_path, "polyglot/lichess_pro_books/lpb-allbook.bin")
    polyglot_reader = chess.polyglot.open_reader(polyglot_path)


    pgn_games = PGNUtils.parse_pgn_file(pgn_path)

    basename_prefix = os.path.basename(pgn_path).replace(".pgn", "")
    boards_tensor, moves_tensor = DatasetUtils.load_dataset_tensor(tensor_folder_path, basename_prefix)

    board_from_tensor = Encoding.board_from_tensor(boards_tensor[0])
    # move_from_tensor = Encoding.moves_from_tensor(moves_tensor[0])

    print(f"bla")
    print(ChessExtra.board_to_string(board_from_tensor))
    pass