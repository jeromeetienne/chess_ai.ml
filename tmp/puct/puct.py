from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import random

class Node:
    def __init__(self, state: Any, parent: Optional[Node] = None):
        self.state: Any = state
        self.parent: Optional[Node] = parent
        self.children: Dict[Any, Node] = {}
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.prior: float = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

class PUCT:
    def __init__(
        self,
        game,
        model=None,
        c_puct: float = 1.0,
        n_simulations: int = 100
    ):
        """
        :param game: A game class instance with methods:
                     - get_legal_moves(state) -> List[Any]
                     - is_terminal(state) -> bool
                     - next_state(state, move) -> Any
                     - current_player(state) -> int
                     - evaluate_state(state) -> float (optional, only if model is None)
        :param model: Optional model with a method predict(state) -> Tuple[Dict[Any,float], float]
                      returning prior probabilities for moves and value of the state
        :param c_puct: Exploration constant
        :param n_simulations: Number of simulations per move
        """
        self.game = game
        self.model = model
        self.c_puct = c_puct
        self.n_simulations = n_simulations

    def search(self, root_state: Any) -> Node:
        root = Node(root_state)
        if self.model:
            priors, _ = self.model.predict(root_state)
            for move, p in priors.items():
                root.children[move] = Node(self.game.next_state(root_state, move), parent=root)
                root.children[move].prior = p
        else:
            for move in self.game.get_legal_moves(root_state):
                root.children[move] = Node(self.game.next_state(root_state, move), parent=root)
                root.children[move].prior = 1.0 / len(root.children)

        for _ in range(self.n_simulations):
            self._simulate(root)

        return root

    def _simulate(self, node: Node) -> float:
        if self.game.is_terminal(node.state):
            # Terminal state, return reward from current player's perspective
            return self.game.evaluate_state(node.state)

        if not node.children:
            # Expand
            moves = self.game.get_legal_moves(node.state)
            for move in moves:
                child_state = self.game.next_state(node.state, move)
                child_node = Node(child_state, parent=node)
                if self.model:
                    priors, _ = self.model.predict(node.state)
                    child_node.prior = priors.get(move, 0.0)
                else:
                    child_node.prior = 1.0 / len(moves)
                node.children[move] = child_node
            return self._evaluate(node)

        # Select child using PUCT formula
        total_visits = sum(child.visits for child in node.children.values())
        best_score = -float('inf')
        best_move = None
        for move, child in node.children.items():
            u = self.c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visits)
            q = child.value
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move

        child = node.children[best_move]
        value = self._simulate(child)

        # Backpropagate
        child.value_sum += value
        child.visits += 1
        return value

    def _evaluate(self, node: Node) -> float:
        if self.model:
            _, value = self.model.predict(node.state)
        else:
            value = self.game.evaluate_state(node.state)
        node.value_sum += value
        node.visits += 1
        return value

    def best_move(self, root: Node) -> Any:
        # Choose the move with highest visit count
        return max(root.children.items(), key=lambda item: item[1].visits)[0]

