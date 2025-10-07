from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
from . gamestate_abc import GameState

class PolicyValueNet(ABC):
    @abstractmethod
    def predict(self, state: GameState) -> Tuple[List[float], float]:
        """
        Given a game state, return:
        - policy: list of prior probabilities for each legal action (in same order as get_legal_actions())
        - value: a float in [-1, 1] (1: win for current player, -1: loss)
        """
        pass
