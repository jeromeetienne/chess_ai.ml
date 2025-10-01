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
    def build_dataset_new(max_files_count: int = 15):

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

    ###############################################################################
    # Create Dataset
    #
    @staticmethod
    def build_dataset(max_files_count: int = 15, max_games_count: int = 7000):
        dataset_creation_start_time = time.time()

        ###############################################################################
        # Load PGN files and parse games
        #
        pgn_file_paths = PGNUtils.all_pgn_file_paths()

        # sort files alphabetically to ensure consistent order
        pgn_file_paths.sort(reverse=False)

        # truncate file_pgn_paths to max_files_count
        pgn_file_paths = pgn_file_paths[:max_files_count]

        games: list[chess.pgn.Game] = []
        for pgn_file_path in pgn_file_paths:
            print(f"Parsing PGN file {os.path.basename(pgn_file_path)} ...", end="", flush=True)
            new_games = PGNUtils.parse_pgn_file(pgn_file_path)  
            games.extend(new_games)
            print(f" {len(new_games)} games")

        ###############################################################################
        # 	 truncate games
        #

        # keep only max_games_count games
        if max_games_count != 0:
            games = games[:max_games_count]

        # keep only the 10 first moves of each game
        slice_game_enabled = False
        if slice_game_enabled:
            sliced_games: list[chess.pgn.Game] = []
            # # opening positions only
            # move_index_start = 0
            # move_index_end = 15
            # middle-game positions only
            move_index_start = 15
            move_index_end = 30
            print(f"Keeping only moves from {move_index_start} to {move_index_end} non included (if possible)")
            for game in games:
                move_count = len(list(game.mainline_moves()))
                # print(f"move_count: {move_count}")
                if move_count < move_index_end:
                    continue
                sliced_game = ChessExtra.game_slice(game, move_index_start, move_index_end)
                sliced_games.append(sliced_game)
            games = list(sliced_games)

        #
        print(f"Game considered: {len(games)}")

        ###############################################################################
        # 	 Create input tensors for the neural network
        #

        # Load polyglot opening book
        polyglot_path = os.path.join(data_folder_path, "polyglot/lichess_pro_books/lpb-allbook.bin")
        polyglot_reader = chess.polyglot.open_reader(polyglot_path)

        # Convert games to tensors
        boards_tensor, moves_tensor = DatasetUtils.games_to_tensor(games, polyglot_reader=polyglot_reader)

        # Save the dataset for later
        DatasetUtils.save_dataset(boards_tensor, moves_tensor, folder_path=output_folder_path)

        # display elapsed time
        dataset_creation_elapsed_time = time.time() - dataset_creation_start_time
        print(f"Dataset creation/loading time: {dataset_creation_elapsed_time:.2f} seconds")

        # Dataset creation stats
        print(DatasetUtils.dataset_summary(boards_tensor, moves_tensor))
