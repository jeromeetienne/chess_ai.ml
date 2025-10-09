import chess
from typing import List, Any
from .gamestate_abc import GameState

class ChessGameState(GameState):
    """GameState adapter for python-chess."""

    def __init__(self, board: chess.Board | None = None):
        self.board = board.copy() if board else chess.Board()

    def clone(self) -> "ChessGameState":
        return ChessGameState(self.board)

    def get_legal_actions(self) -> List[chess.Move]:
        """Return all legal moves as a list of chess.Move objects."""
        legal_moves = list(self.board.legal_moves)
        return legal_moves

    def apply_action(self, action: chess.Move) -> "ChessGameState":
        """Apply a move and return a new game state."""
        new_board = self.board.copy()
        new_board.push(action)
        return ChessGameState(new_board)

    def is_terminal(self) -> bool:
        """True if the game is over (checkmate, stalemate, draw, etc.)."""
        return self.board.is_game_over()

    def get_result(self) -> float:
        """
        Returns result from perspective of current player (before the move):
        - 1.0 if current player wins
        - 0.0 if loses
        - 0.5 if draw
        """
        result = self.board.result(claim_draw=True)
        if result == "1-0":
            return 1.0 if self.board.turn == chess.WHITE else 0.0
        elif result == "0-1":
            return 1.0 if self.board.turn == chess.BLACK else 0.0
        else:
            return 0.5

    def current_player(self) -> int:
        """Return +1 for White to move, -1 for Black."""
        return 1 if self.board.turn == chess.WHITE else -1

    def __str__(self) -> str:
        return str(self.board)
