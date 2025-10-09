from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
from .gamestate_abc import GameState


class PolicyValueNet(ABC):
    @abstractmethod
    def predict(self, state: GameState) -> Tuple[List[float], float]:
        """
        Given a game state, return:
        - policy: list of prior probabilities for each legal action (in same order as get_legal_actions())
        - value: a float in [-1, 1] (1: win for current player, -1: loss)
        """
        pass

    def predict_batch(self, states: List[GameState]) -> Tuple[List[List[float]], List[float]]:
        """
        Optional batched prediction helper. Default implementation just calls
        `predict` for each state sequentially. Subclasses that can perform
        efficient batched inference should override this method.

        Parameters
        ----------
        states: List[GameState]
            A list of game states to evaluate in batch.

        Returns
        -------
        Tuple[List[List[float]], List[float]]
            A tuple (priors_list, values_list) where priors_list[i] is the
            list of prior probabilities for the i-th state's legal moves and
            values_list[i] is the scalar value for the i-th state.
        """
        priors_list: List[List[float]] = []
        values_list: List[float] = []
        for s in states:
            priors, val = self.predict(s)
            priors_list.append(priors)
            values_list.append(val)
        return priors_list, values_list
