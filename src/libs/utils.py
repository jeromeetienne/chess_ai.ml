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
        max_files_count = 25
        # max_files_count = 18
        # max_files_count = 10
        pgn_file_paths = pgn_file_paths[:max_files_count]

        games: list[chess.pgn.Game] = []
        for file_index, pgn_file_path in enumerate(pgn_file_paths):
            print(f"processing file {pgn_file_path} ({file_index+1}/{len(pgn_file_paths)})")
            new_games = PGNUtils.load_games_from_pgn(f"{pgn_folder_path}/{pgn_file_path}")
            games.extend(new_games)
            print(f"GAMES LOADED: {len(games)}")

        ###############################################################################
        ###############################################################################
        #	 Shuffle and truncate games
        ###############################################################################
        ###############################################################################
                
        # Shuffle the games
        # random_seed = 42
        # torch.manual_seed(random_seed)
        # games_rnd_indexes = torch.randperm(len(games)).tolist()
        # games = [games[i] for i in games_rnd_indexes]

        # keep only max_games_count games
        max_games_count = len(games)
        # max_games_count = 7_000
        # max_games_count = 1_000
        # max_games_count = 100
        games = games[:max_games_count]
        #
        print(f"GAMES PARSED: {len(games)}")

        ###############################################################################
        ###############################################################################
        #	 Create input tensors for the neural network
        ###############################################################################
        ###############################################################################

        boards_tensor, best_move_tensor = EncodingUtils.create_input_for_nn(games)

        # # Truncate to 2.5 million samples
        # boards_tensor = boards_tensor[0:2500000]
        # best_move_tensor = best_move_tensor[0:2500000]

        # Encode moves
        best_move_tensor, uci_to_classindex = EncodingUtils.encode_moves(best_move_tensor)

        # Convert to PyTorch tensors
        boards_tensor = torch.tensor(boards_tensor, dtype=torch.float32)
        best_move_tensor = torch.tensor(best_move_tensor, dtype=torch.long)

        # print dataset stats
        return boards_tensor, best_move_tensor, uci_to_classindex
