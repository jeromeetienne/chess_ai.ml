from ..games.game_tictactoe import GameTicTacToe
from typing import List, Optional, Protocol, runtime_checkable

@runtime_checkable
class PlayerProtocol(Protocol):
    """
    Protocol/Interface for any TicTacToe player (Human or AI).
    A player must have a player_id (1 or -1) and a get_move method.
    """
    player_id: int
    marker: str

    def get_move(self, game: GameTicTacToe) -> int:
        """
        Calculates and returns the player's chosen move (index 0-8).
        """
        ...
