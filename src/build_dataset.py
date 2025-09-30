# stdlib imports
import os
import time

# pip imports
import chess
import chess.pgn
import chess.polyglot


# local imports
from .libs.chess_extra import ChessExtra
from .libs.encoding import Encoding
from .libs.utils import Utils
from .libs.io_dataset import IoDataset

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.join(__dirname__, "..", "output")
data_folder_path = os.path.join(__dirname__, "..", "data")
class DatasetBuilderCommand:
    ###############################################################################
    # Create Dataset
    #
    @staticmethod
    def build_dataset(max_files_count: int = 15, max_games_count: int = 7000):
        dataset_creation_start_time = time.time()

        ###############################################################################
        # Load PGN files and parse games
        #
        pgn_folder_path = os.path.join(data_folder_path, "./pgn")
        pgn_basenames = [file_path for file_path in os.listdir(pgn_folder_path) if file_path.endswith(".pgn")]

        # sort files alphabetically to ensure consistent order
        pgn_basenames.sort(reverse=False)

        # truncate file_pgn_paths to max_files_count
        pgn_basenames = pgn_basenames[:max_files_count]

        games: list[chess.pgn.Game] = []
        for pgn_basename in pgn_basenames:
            pgn_file_path = os.path.join(pgn_folder_path, pgn_basename)
            print(f"Parsing PGN file {pgn_basename} ...", end="", flush=True)
            new_games: list[chess.pgn.Game] = []
            with open(pgn_file_path, 'r') as pgn_file:
                while True:
                    if max_games_count != 0 and len(games) >= max_games_count:
                        break
                    game: chess.pgn.Game|None = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    new_games.append(game)
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
        boards_tensor, moves_tensor = Encoding.games_to_tensor(games, polyglot_reader=polyglot_reader)

        # Save the dataset for later
        IoDataset.save_dataset(boards_tensor, moves_tensor, folder_path=output_folder_path)

        # display elapsed time
        dataset_creation_elapsed_time = time.time() - dataset_creation_start_time
        print(f"Dataset creation/loading time: {dataset_creation_elapsed_time:.2f} seconds")

        # Dataset creation stats
        print(Utils.dataset_summary(boards_tensor, moves_tensor))
