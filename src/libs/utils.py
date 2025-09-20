# stdlib imports
import os

# pip imports
import torch
import chess.pgn

# local imports
from libs.pgn_utils import PGNUtils
from libs.encoding_utils import EncodingUtils

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))

class Utils:
    @staticmethod
    def create_dataset() -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        print("Creating dataset...")
        ###############################################################################
        # Load PGN files and parse games
        #
        pgn_folder_path = f"{__dirname__}/../../data/pgn"
        pgn_file_paths = [file for file in os.listdir(pgn_folder_path) if file.endswith(".pgn")]
        # sort files alphabetically to ensure consistent order
        pgn_file_paths.sort(reverse=False)

        # truncate file_pgn_paths to max_files_count
        max_files_count = 28
        # max_files_count = 22
        # max_files_count = 25
        max_files_count = 18
        max_files_count = 10
        pgn_file_paths = pgn_file_paths[:max_files_count]


        games: list[chess.pgn.Game] = []
        for file_index, pgn_file_path in enumerate(pgn_file_paths):
            print(f"processing file {pgn_file_path} ({file_index+1}/{len(pgn_file_paths)})")
            new_games = PGNUtils.load_games_from_pgn(f"{pgn_folder_path}/{pgn_file_path}")
            games.extend(new_games)
            print(f"GAMES LOADED: {len(games)}")

        # Shuffle the games
        # random_seed = 42
        # torch.manual_seed(random_seed)
        # games_rnd_indexes = torch.randperm(len(games)).tolist()
        # games = [games[i] for i in games_rnd_indexes]

        # keep only max_games_count games
        max_games_count = len(games)
        # max_games_count = 7_000
        # max_games_count = 1_000
        max_games_count = 100
        games = games[:max_games_count]
        #
        print(f"GAMES PARSED: {len(games)}")


        ###############################################################################
        # Convert data into tensors
        #
        X, y = EncodingUtils.create_input_for_nn(games)

        print(f"NUMBER OF SAMPLES: {len(y)}")

        # Truncate to 2.5 million samples
        X = X[0:2500000]
        y = y[0:2500000]

        # Encode moves
        y, uci_to_classindex = EncodingUtils.encode_moves(y)


        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return X, y,uci_to_classindex