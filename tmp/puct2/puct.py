from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
import math
from .gamestate_abc import GameState
from .policyvaluenet_abc import PolicyValueNet

class PUCTNode:
    def __init__(self, parent: Optional[PUCTNode], prior: float):
        self.parent = parent
        self.children: dict[Any, PUCTNode] = {}
        self.prior = prior

        self.visit_count = 0
        self.value_sum = 0.0

    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class PUCT:
    def __init__(self, policy_value_fn: PolicyValueNet, c_puct: float = 1.4):
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct

    def search(self, root_state: GameState, num_simulations: int = 800) -> Any:
        root = PUCTNode(parent=None, prior=1.0)

        # expand root
        self._expand(root, root_state)

        for _ in range(num_simulations):
            node, state = self._traverse(root, root_state.clone())
            value = self._evaluate(node, state)
            self._backpropagate(node, value)

        # pick the action with max visit count
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    def _traverse(self, node: PUCTNode, state: GameState) -> Tuple[PUCTNode, GameState]:
        """Traverse the tree using PUCT selection until a leaf is reached."""
        while node.is_expanded() and not state.is_terminal():
            action, node = self._select_child(node)
            state = state.apply_action(action)
        return node, state

    def _select_child(self, node: PUCTNode) -> Tuple[Any, PUCTNode]:
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score = -float('inf')
        best_action, best_child = None, None

        for action, child in node.children.items():
            q_value = -child.value()  # flip sign because perspective alternates
            u_value = self.c_puct * child.prior * math.sqrt(total_visits + 1) / (1 + child.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action, best_child = action, child

        return best_action, best_child

    def _expand(self, node: PUCTNode, state: GameState) -> None:
        if state.is_terminal():
            return
        legal_actions = state.get_legal_actions()
        priors, _ = self.policy_value_fn.predict(state)

        for action, prior in zip(legal_actions, priors):
            node.children[action] = PUCTNode(parent=node, prior=prior)

    def _evaluate(self, node: PUCTNode, state: GameState) -> float:
        if state.is_terminal():
            return state.get_result()
        self._expand(node, state)
        _, value = self.policy_value_fn.predict(state)
        return value

    def _backpropagate(self, node: PUCTNode, value: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # switch perspective for parent
            node = node.parent
