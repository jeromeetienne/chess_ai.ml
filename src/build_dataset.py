# stdlib imports
import os
import time

# pip imports
import chess
import chess.pgn
import chess.polyglot
import torch


# local imports
from .libs.chess_extra import ChessExtra
from .libs.encoding import Encoding
from .utils.dataset_utils import DatasetUtils
from .utils.pgn_utils import PGNUtils

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.join(__dirname__, "..", "output")
data_folder_path = os.path.join(__dirname__, "..", "data")
tensors_folder_path = os.path.join(output_folder_path, "pgn_tensors")


class DatasetBuilderCommand:
    ###############################################################################
    # Create Dataset
    #
    @staticmethod
    def build_dataset(max_files_count: int = 15):

        ###############################################################################
        # Load PGN files and parse games
        #
        pgn_file_paths = PGNUtils.get_pgn_paths()

        # truncate file_pgn_paths to max_files_count
        if max_files_count > 0:
            pgn_file_paths = pgn_file_paths[:max_files_count]

        ###############################################################################
        #   Load polyglot opening book
        #
        # Load polyglot opening book
        polyglot_path = os.path.join(data_folder_path, "polyglot/lichess_pro_books/lpb-allbook.bin")
        polyglot_reader = chess.polyglot.open_reader(polyglot_path)

        ###############################################################################
        # 	 Create input tensors for the neural network
        #

        for pgn_file_path in pgn_file_paths:
            basename = os.path.basename(pgn_file_path).replace(".pgn", "")
            boards_file_path = os.path.join(tensors_folder_path, f"{basename}{DatasetUtils.FILE_SUFFIX.BOARDS}")
            moves_file_path = os.path.join(tensors_folder_path, f"{basename}{DatasetUtils.FILE_SUFFIX.MOVES}")


            # Skip if files already exist
            if os.path.exists(boards_file_path) and os.path.exists(moves_file_path):
                print(f"{basename}.pgn already got a boards and moves tensor files, skipping.")
                continue


            time_start = time.perf_counter()

            print(f"{basename}.pgn converting to tensors ... ", end="", flush=True)
            games = PGNUtils.pgn_file_to_games(pgn_file_path)

            # split the games into boards and moves
            boards, moves = DatasetUtils.games_to_boards_moves(games, polyglot_reader)
            # convert the boards and moves to tensors
            boards_tensor, moves_tensor = DatasetUtils.boards_moves_to_tensor(boards, moves)

            # Save boards and moves tensors
            torch.save(boards_tensor, boards_file_path)
            torch.save(moves_tensor, moves_file_path)

            time_elapsed = time.perf_counter() - time_start
            print(f"Done. {str(boards_tensor.shape[0]).rjust(5)} boards in {time_elapsed:.2f} seconds, avg {boards_tensor.shape[0]/time_elapsed:.2f} boards/sec ")

