# stdlib imports
import os
import random

# pip imports
import torch
import chess.pgn
import numpy as np

# local imports
from libs.chess_extra import ChessExtra
from libs.pgn_utils import PGNUtils
from libs.encoding_utils import EncodingUtils
from libs.chess_model import ChessModel

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
            print(f"processing file {pgn_file_path} ({file_index+1}/{len(pgn_file_paths)})")
            new_games = PGNUtils.load_games_from_pgn(f"{pgn_folder_path}/{pgn_file_path}")
            games.extend(new_games)
            print(f"GAMES LOADED: {len(games)}")

        ###############################################################################
        ###############################################################################
        # 	 Shuffle and truncate games
        ###############################################################################
        ###############################################################################

        # keep only max_games_count games
        if max_games_count != 0:
            games = games[:max_games_count]
        #
        print(f"GAMES PARSED: {len(games)}")

        # keep only the 10 first moves of each game
        if False:
            sliced_games: list[chess.pgn.Game] = []
            move_index_start = 0
            move_index_end = 10
            for game in games:
                move_count = len(list(game.mainline_moves()))
                # print(f"move_count: {move_count}")
                if move_count < move_index_end:
                    continue
                sliced_game = ChessExtra.game_slice(game, move_index_start, move_index_end)
                sliced_games.append(sliced_game)
            games = list(sliced_games)

        ###############################################################################
        ###############################################################################
        # 	 Create input tensors for the neural network
        ###############################################################################
        ###############################################################################

        boards_nparray, best_move_nparray, uci_to_classindex = EncodingUtils.create_input_for_nn_np(games)

        # Encode moves
        # best_move_nparray, uci_to_classindex = EncodingUtils.encode_moves_np(best_move_nparray)

        # Convert to PyTorch tensors
        boards_tensor = torch.tensor(boards_nparray, dtype=torch.float32)
        best_move_tensor = torch.tensor(best_move_nparray, dtype=torch.long)

        # print dataset stats
        return boards_tensor, best_move_tensor, uci_to_classindex

    @staticmethod
    def select_best_move(
        board: chess.Board, probabilities: np.ndarray, classindex_to_uci: dict[int, str], random_threshold: float = 1
    ) -> str | None:
        """
        Select the best move based on the model's output probabilities.

        Args:
            board (chess.Board): The current state of the chess board.
            probabilities (np.ndarray): The output probabilities from the model.
            classindex_to_uci (dict[int, str]): Mapping from class indices to UCI move strings.
            random_threshold (float): Threshold to introduce randomness in move selection.
                                      If 1, selects the best move. If less than 1, allows for some
                                      randomness among top moves.
        Returns:
            str | None: The selected move in UCI format, or None if no legal move is found.
        """
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probabilities = probabilities[sorted_indices]

        # TODO make the possibility to return top N moves with their probabilities, not just the best one
        # - thus allowing to implement some randomness in the move selection
        proposed_top_n = 5
        proposed_moves_uci_proba = []
        for classindex in sorted_indices:
            move_uci = classindex_to_uci[classindex]
            # skip illegal moves
            if move_uci not in legal_moves_uci:
                continue

            proposed_moves_uci_proba.append((move_uci, probabilities[classindex]))
            if len(proposed_moves_uci_proba) >= proposed_top_n:  # Get top 5 legal moves
                break

        # if no legal move found (should not happen in a normal game)
        if len(proposed_moves_uci_proba) == 0:
            return None  # No legal moves found (should not happen in a normal game)

        # keep only the proposed moves with a probability close to the best one
        if len(proposed_moves_uci_proba) > 1:
            best_proba = proposed_moves_uci_proba[0][1]
            threshold = random_threshold * best_proba  # Keep moves with at least 80% of the best probability
            proposed_moves_uci_proba = [(move, proba) for move, proba in proposed_moves_uci_proba if proba >= threshold]

        # sanity check
        if random_threshold == 1.0:
            assert len(proposed_moves_uci_proba) <= 1, "should be at most 1 move after thresholding"

        # return any of the proposed moves randomly
        best_move_uci = random.choice(proposed_moves_uci_proba)[0]

        return best_move_uci

    @staticmethod
    def predict_next_move(
        board: chess.Board, model: ChessModel, device: str, classindex_to_uci: dict[int, str]
    ) -> str | None:
        """
        Predict the best move for the given board state.
        Args:
            board (chess.Board): The current state of the chess board.
        Returns:
            str | None: The predicted best move in UCI format, or None if no legal move is found.
        """
        boards_tensor = EncodingUtils.board_to_tensor(board).to(device)

        # Set the model to evaluation mode (it may be reductant)
        model.eval()

        # Disable gradient calculation for inference
        with torch.no_grad():
            logits = model(boards_tensor)

        logits = logits.squeeze(0)  # Remove batch dimension

        probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities

        best_move_uci = Utils.select_best_move(board, probabilities, classindex_to_uci)
        return best_move_uci
