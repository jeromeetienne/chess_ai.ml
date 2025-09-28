# stdlib imports
import os

# pip imports
import torch
import chess.pgn

# local imports
from .chess_extra import ChessExtra
from .pgn_utils import PGNUtils
from .encoding import Encoding

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))


class Utils:
    @staticmethod
    def create_dataset(
        max_files_count: int = 15, max_games_count: int = 7000
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        print("Creating dataset...")

        ###############################################################################
        # Load PGN files and parse games
        #
        pgn_folder_path = f"{__dirname__}/../../data/pgn"
        pgn_file_paths = [file for file in os.listdir(pgn_folder_path) if file.endswith(".pgn")]

        # sort files alphabetically to ensure consistent order
        pgn_file_paths.sort(reverse=False)

        # truncate file_pgn_paths to max_files_count
        pgn_file_paths = pgn_file_paths[:max_files_count]

        games: list[chess.pgn.Game] = []
        for file_index, pgn_file_path in enumerate(pgn_file_paths):
            print(f"Parsing PGN file {pgn_file_path} ...", end="")
            new_games = PGNUtils.load_games_from_pgn(f"{pgn_folder_path}/{pgn_file_path}")
            games.extend(new_games)
            print(f" {len(new_games)} games")

        ###############################################################################
        ###############################################################################
        # 	 Shuffle and truncate games
        ###############################################################################
        ###############################################################################

        # keep only max_games_count games
        if max_games_count != 0:
            games = games[:max_games_count]

        # keep only the 10 first moves of each game
        if False:
            sliced_games: list[chess.pgn.Game] = []
            # # opening positions only
            # move_index_start = 0
            # move_index_end = 15
            # middle-game positions only
            # move_index_start = 15
            # move_index_end = 30
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
        ###############################################################################
        # 	 Create input tensors for the neural network
        ###############################################################################
        ###############################################################################

        boards_tensor, moves_tensor, uci_to_classindex = Encoding.games_to_tensor(games)


        # print dataset stats
        return boards_tensor, moves_tensor, uci_to_classindex

    @staticmethod
    def model_summary(model: torch.nn.Module) -> str:
        """
        Prints a basic summary of the model including parameter count.
        """
        total_params = 0
        trainable_params = 0
        
        output = []
        output.append(f"\nModel Architecture: {model.__class__.__name__}")
        output.append("-" * 60)
        output.append(f"{'Layer Name':<30} {'Param Count':>15} {'Trainable':>10}")
        output.append("=" * 60)
        
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue    
            
            param = parameter.numel()
            total_params += param
            
            if parameter.requires_grad:
                trainable_params += param
            
            output.append(f"{name:<30} {param:>15,} {'Yes' if parameter.requires_grad else 'No':>10}")

        output.append("=" * 60)
        output.append(f"Total Parameters: {total_params:,}")
        output.append(f"Trainable Parameters: {trainable_params:,}")
        output.append(f"Non-trainable Parameters: {total_params - trainable_params:,}")
        output.append("-" * 60)

        return "\n".join(output)

    @staticmethod
    def dataset_summary(boards_tensor: torch.Tensor, moves_tensor: torch.Tensor, uci_to_classindex: dict) -> str:
            summary = f"""Dataset Summary:
- Total positions: {len(boards_tensor):,}
- Input shape: {boards_tensor.shape[1:]} (Channels, Height, Width)
- Output shape: {moves_tensor.shape[1:]} (Scalar class index)
- Number of unique moves (classes): {len(uci_to_classindex):,}
- Sample move index (first position): {moves_tensor[0].item()}
- Sample move UCI (first position): {list(uci_to_classindex.keys())[list(uci_to_classindex.values()).index(moves_tensor[0].item())]}
"""
            return summary

