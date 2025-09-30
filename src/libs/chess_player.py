# stdlib imports
import random
import os

# pip imports
import torch
import chess
import chess.polyglot
import numpy as np

from src.libs.uci2class_utils import Uci2ClassUtils


# local imports
from .encoding import Encoding
from .chess_model import ChessModel

__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "../../data")

class ChessPlayer:
    def __init__(self, model: ChessModel, color: chess.Color , polyglot_reader: chess.polyglot.MemoryMappedReader|None = None):
        """
        Initialize the ChessbotMLPlayer with the given model, class index to UCI mapping, and optional polyglot reader.
        """
        self._color = color
        self._model = model
        self._class2uci = Uci2ClassUtils.get_class2uci(color)
        self._polyglot_reader = polyglot_reader

    def predict_next_move(self, board: chess.Board) -> str | None:
        """
        Predict the best move for the given board state.
        Args:
            board (chess.Board): The current state of the chess board.
        Returns:
            str | None: The predicted best move in UCI format, or None if no legal move is found.
        """
        # First try to get the move from the opening book
        best_move_uci = self._predict_next_move_opening_book(board)
        if best_move_uci is not None:
            return best_move_uci

        # If no opening move is found, use the ML model to predict the move
        best_move_uci = self._predict_next_move_ml(board)
        return best_move_uci

    ###############################################################################
    #   opening book based move prediction
    #
    def _predict_next_move_opening_book(self, board: chess.Board) -> str | None:
        """
        Predict the best move for the given board state using the opening book.
        Args:
            board (chess.Board): The current state of the chess board.
        Returns:
            str | None: The predicted best move in UCI format, or None if no opening move is found.
        """
        move_uci: str | None = None

        if self._polyglot_reader is None:
            return None

        try:
            opening_entry = self._polyglot_reader.weighted_choice(board)
            move_uci = opening_entry.move.uci()
        except IndexError:
            move_uci = None  # No opening entry found.

        return move_uci

    ###############################################################################
    #   machine learning based move prediction
    #
    def _predict_next_move_ml(self, board: chess.Board) -> str | None:
        """
        Predict the best move for the given board state using the ML model.
        Args:
            board (chess.Board): The current state of the chess board.
        Returns:
            str | None: The predicted best move in UCI format, or None if no legal move is found.
        """
        # Check for GPU
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
        # print(f"Using device: {device}")

        boards_tensor = Encoding.board_to_tensor(board).unsqueeze(0).to(device)

        # Set the model to evaluation mode
        self._model.eval()

        # move the model to the device
        self._model.to(device)

        # Disable gradient calculation for inference
        with torch.no_grad():
            logits: torch.Tensor = self._model(boards_tensor)

        logits = logits.squeeze(0)  # Remove batch dimension

        probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities

        best_move_uci = self._select_best_move_ml(board, probabilities)
        return best_move_uci

    def _select_best_move_ml(self, board: chess.Board, probabilities: np.ndarray, random_threshold: float = 0.95) -> str | None:
        """
        Select the best move based on the model's output probabilities. only for model-based move prediction.

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
            move_uci = self._class2uci[classindex]
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

