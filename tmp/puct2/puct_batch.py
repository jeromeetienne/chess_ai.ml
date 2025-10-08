from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
import math
from .gamestate_abc import GameState
from .policyvaluenet_abc import PolicyValueNet


class PUCTNode:
    parent: Optional[PUCTNode]
    """Parent node in the tree (None for the root)."""

    children: dict[Any, PUCTNode]
    """Mapping from actions to child nodes."""

    prior: float
    """Prior probability P(s, a) returned by the policy network for the
    action that led to this node."""

    visit_count: int
    """Number of times this node has been visited during MCTS."""

    value_sum: float
    """Sum of the backed-up value estimates for this node; divide by
    `visit_count` to obtain the average value estimate."""

    def __init__(self, parent: Optional[PUCTNode], prior: float):
        self.parent = parent
        self.children: dict[Any, PUCTNode] = {}
        self.prior = prior

        self.visit_count = 0
        self.value_sum = 0.0

    def value(self) -> float:
        """
        Return the mean value estimate for this node.

        If the node has not been visited yet (`visit_count == 0`) this returns
        0.0 as a neutral default. Otherwise it returns `value_sum / visit_count`,
        the average of all backed-up values accumulated for this node.

        Returns
        -------
        float
            The average value for the node (or 0.0 if unvisited). Typical
            ranges depend on the evaluation source (for example -1..1).

        Complexity
        ----------
        O(1)
        """
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """
        Return True if this node has been expanded (i.e., it has children).

        This is a quick check used by the search traversal to determine
        whether selection should continue down this node or stop at a leaf.

        Returns
        -------
        bool
            True if `node.children` is non-empty, False otherwise.

        Complexity
        ----------
        O(1) — checks the length of the children dict.
        """
        return len(self.children) > 0


class PUCT:
    def __init__(self, policy_value_fn: PolicyValueNet, c_puct: float = 1.4):
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct

    def search(self, root_state: GameState, num_simulations: int = 800) -> Any:
        """
        Run PUCT search from `root_state` and return the chosen action.

        The algorithm initializes a single root `PUCTNode`, expands it using the
        policy-value network, then performs `num_simulations` Monte Carlo tree
        search simulations. Each simulation traverses the current tree using
        `_traverse`, evaluates the reached leaf with `_evaluate`, and updates
        statistics via `_backpropagate`.

        Parameters
        ----------
        root_state: GameState
            The game state at the root of the search. This should represent the
            current position for which we want to pick an action.
        num_simulations: int
            Number of MCTS simulations to run. More simulations yield more
            accurate visit counts but take more time.

        Returns
        -------
        Any
            The selected action (the key from `root.children`) with the highest
            visit count after all simulations. The return type depends on the
            `GameState` implementation's action representation.

        Side effects
        ------------
        - Populates and mutates the search tree rooted at the created `root`
          node (visit counts, value sums, and child nodes are modified).
        - Does not modify the provided `root_state` thanks to `root_state.clone()`
          used during simulations.
        """
        root = PUCTNode(parent=None, prior=1.0)

        # expand root
        # Use the policy/value network to expand root's children
        self._expand(root, root_state)

        # We'll collect leaf nodes from traversals and evaluate them in
        # batches using policy_value_fn.predict_batch when possible.
        batch_size = 64
        sims_done = 0
        while sims_done < num_simulations:
            batch_leaves: List[Tuple[PUCTNode, GameState]] = []
            # collect up to batch_size leaves
            for _ in range(min(batch_size, num_simulations - sims_done)):
                node, state = self._traverse(root, root_state.clone())
                batch_leaves.append((node, state))

            # prepare states for batch prediction
            states = [s for (_n, s) in batch_leaves]

            # call predict_batch (fall back to sequential predict_batch default)
            priors_list, values_list = self.policy_value_fn.predict_batch(states)

            # expand nodes and backpropagate values
            for (node, state), priors, value in zip(batch_leaves, priors_list, values_list):
                # if terminal, use terminal result instead of model value
                if state.is_terminal():
                    value = state.get_result()
                else:
                    # expand using priors returned by the model; priors correspond
                    # to legal moves in state.get_legal_actions() order
                    self._expand_with_priors(node, state, priors)

                self._backpropagate(node, value)

            sims_done += len(batch_leaves)

        # pick the action with max visit count
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    def _traverse(self, node: PUCTNode, state: GameState) -> Tuple[PUCTNode, GameState]:
        """
        Traverse the search tree from `node` following PUCT selection until a
        leaf node or a terminal state is reached.

        The method repeatedly selects a child using `_select_child` and applies
        the chosen action to `state` (via `state.apply_action`) until either
        the current node has no children (is a leaf) or `state` is terminal.

        Parameters
        ----------
        node: PUCTNode
            The starting node in the tree.
        state: GameState
            The game state corresponding to `node`. This object will be
            mutated by successive calls to `state.apply_action`; callers should
            pass a cloned copy when they want to preserve the original state.

        Returns
        -------
        Tuple[PUCTNode, GameState]
            A pair (leaf_node, leaf_state) where `leaf_node` is the final node
            reached (either unexpanded or corresponding to a terminal state)
            and `leaf_state` is the game state after applying the sequence of
            actions taken during traversal.
        """
        while node.is_expanded() and not state.is_terminal():
            action, node = self._select_child(node)
            state = state.apply_action(action)
        return node, state

    def _select_child(self, node: PUCTNode) -> Tuple[Any, PUCTNode]:
        """
        Select the child of `node` with the highest PUCT score.

        The PUCT score used is:

                score = Q(s, a) + c_puct * P(s, a) * sqrt(N_parent + 1) / (1 + N(s, a))

        where:
        - Q(s, a) is the negative of the child's value() because we flip
            perspective between parent and child.
        - P(s, a) is the prior probability stored in the child node.
        - N_parent is the total visits to all children of `node`.
        - N(s, a) is the visit count of the child.

        Parameters
        ----------
        node: PUCTNode
                The parent node whose children will be scored and from which a
                single (action, child) pair is returned.

        Returns
        -------
        Tuple[Any, PUCTNode]
                The selected (action, child) pair. The method asserts that at
                least one child exists and will raise an AssertionError otherwise.

        Notes
        -----
        - If there are more actions than priors (or vice versa) elsewhere in
            the code, selection here relies on the children already present on
            `node` and scores whatever entries exist in `node.children`.
        - Ties are broken by the > operator on computed scores; the first
            child with the maximal score encountered will be returned.
        """
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score = -float("inf")
        best_action, best_child = None, None

        for action, child in node.children.items():
            q_value = -child.value()  # flip sign because perspective alternates
            u_value = self.c_puct * child.prior * math.sqrt(total_visits + 1) / (1 + child.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action, best_child = action, child

        assert best_action is not None and best_child is not None

        return best_action, best_child

    def _expand(self, node: PUCTNode, state: GameState) -> None:
        """
        Expand `node` by creating child nodes for each legal action in `state`.

        If `state` is terminal this is a no-op. Otherwise this method queries the
        policy-value network to obtain prior probabilities for legal moves and
        creates a new `PUCTNode` for each (action, prior) pair, storing them in
        `node.children` keyed by the action.

        Parameters
        ----------
        node: PUCTNode
            The tree node that corresponds to `state`. New children will have
            `node` set as their `parent`.
        state: GameState
            The game state to expand. Must match the position represented by
            `node` (caller responsibility).

        Side effects
        ------------
        - Populates `node.children` with entries for each legal action.
        - Does not modify `state`.

        Notes
        -----
        - The function zips `legal_actions` with the priors returned by the
          policy-value network; if the arrays differ in length, `zip` will
          silently truncate to the shorter one.
        """
        if state.is_terminal():
            return
        legal_actions = state.get_legal_actions()
        priors, _ = self.policy_value_fn.predict(state)

        for action, prior in zip(legal_actions, priors):
            node.children[action] = PUCTNode(parent=node, prior=prior)

    def _expand_with_priors(self, node: PUCTNode, state: GameState, priors: List[float]) -> None:
        """
        Expand `node` using a provided list of priors that correspond to
        `state.get_legal_actions()` order. This avoids calling the model
        again when priors are already available (e.g., from a batched
        prediction).
        """
        if state.is_terminal():
            return
        legal_actions = state.get_legal_actions()
        for action, prior in zip(legal_actions, priors):
            node.children[action] = PUCTNode(parent=node, prior=prior)

    def _evaluate(self, node: PUCTNode, state: GameState) -> float:
        """
        Evaluate a leaf state and return a value from the perspective of the
        player to move in `state`.

        Behavior:
        - If `state` is terminal, returns `state.get_result()` (game outcome).
        - Otherwise, expands `node` (creating children for legal moves), queries
          the policy-value network for a (priors, value) pair and returns the
          `value` component.

        Parameters
        ----------
        node: PUCTNode
            The node corresponding to `state`. This node will be expanded when
            `state` is non-terminal.
        state: GameState
            The game state to evaluate.

        Returns
        -------
        float
            A scalar value estimate for `state` from the perspective of the
            player to move (typically in a range like -1..1 depending on the
            policy-value network).
        """
        # This method is kept for backward compatibility but in the batched
        # search flow we prefer to use predict_batch and _expand_with_priors.
        if state.is_terminal():
            return state.get_result()
        self._expand(node, state)
        _, value = self.policy_value_fn.predict(state)
        return value

    def _backpropagate(self, node: Optional[PUCTNode], value: float) -> None:
        """
        Backpropagate a value up the tree starting from `node` and ending at the root.

        This updates each visited node's `visit_count` and `value_sum`.

        The provided `value` is assumed to be from the perspective of `node` (i.e.
        a positive value is good for the player to move at `node`). Because the
        player to move alternates between parent and child, the sign of `value`
        is flipped before adding it to the parent's `value_sum`.

        Parameters
        ----------
        node: PUCTNode
            The starting node for backpropagation. If `None`, the method does nothing.
        value: float
            The value to backpropagate (model or game-evaluation output). Typical
            ranges depend on the evaluation source (for example -1..1), but any
            float is accepted.

        Side effects
        ------------
        - Increments `visit_count` for every node on the path from `node` to the root.
        - Adds the (possibly sign-flipped) `value` to each node's `value_sum`.
        """
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # switch perspective for parent
            node = node.parent
