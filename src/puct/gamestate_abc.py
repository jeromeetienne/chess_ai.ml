from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional

class GameState(ABC):
    """Abstract representation of a game state for a 2-player turn-based game."""

    @abstractmethod
    def clone(self) -> GameState:
        """Return a deep copy of the current game state."""
        pass

    @abstractmethod
    def get_legal_actions(self) -> List[Any]:
        """Return a list of all legal actions from this state."""
        pass

    @abstractmethod
    def apply_action(self, action: Any) -> GameState:
        """Return a new GameState after applying the given action."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the game has ended."""
        pass

    @abstractmethod
    def get_result(self) -> float:
        """
        Return the result from the perspective of the current player:
        - 1.0 if current player wins
        - 0.0 if loss
        - 0.5 if draw
        """
        pass

    @abstractmethod
    def current_player(self) -> int:
        """Return 1 or -1 indicating whose turn it is."""
        pass
