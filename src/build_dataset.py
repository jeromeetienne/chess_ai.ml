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
tensors_folder_path = os.path.join(__dirname__, "..", "data", "pgn_tensors")


class DatasetBuilderCommand:
    ###############################################################################
    # Create Dataset
    #
    @staticmethod
    def build_dataset(max_files_count: int = 15):

        ###############################################################################
        # Load PGN files and parse games
        #
        pgn_file_paths = PGNUtils.all_pgn_file_paths()

        # sort files alphabetically to ensure consistent order
        pgn_file_paths.sort()

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
            boards_file_path = os.path.join(tensors_folder_path, f"{basename}_boards_tensor.pt")
            moves_file_path = os.path.join(tensors_folder_path, f"{basename}_moves_tensor.pt")


            # Skip if files already exist
            if os.path.exists(boards_file_path) and os.path.exists(moves_file_path):
                print(f"{basename}.pgn already got a boards and moves tensor files, skipping.")
                continue


            time_start = time.perf_counter()

            print(f"{basename}.pgn converting to tensors ... ", end="", flush=True)
            games = PGNUtils.parse_pgn_file(pgn_file_path)

            # Convert games to tensors
            boards_tensor, moves_tensor = DatasetUtils.games_to_tensor(games, polyglot_reader=polyglot_reader)

            # Save boards and moves tensors
            torch.save(boards_tensor, boards_file_path)
            torch.save(moves_tensor, moves_file_path)

            time_elapsed = time.perf_counter() - time_start
            print(f"Done. {str(boards_tensor.shape[0]).rjust(5)} boards in {time_elapsed:.2f} seconds, avg {boards_tensor.shape[0]/time_elapsed:.2f} boards/sec ")

