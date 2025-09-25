from .tictactoe import TicTacToe
import random
from .mtcs_node import MCTSNode
import typing

class MCTSPlayer:
    """
    An AI player that uses Monte Carlo Tree Search to determine the best move.
    """
    def __init__(self, player_id: int, simulations: int = 1000, c_param: float = 1.4):
        self.player_id: int = player_id
        self.marker: str = 'X' if player_id == 1 else 'O'
        self.simulations: int = simulations
        self.c_param: float = c_param # Exploration constant for UCT

    def get_move(self, game: TicTacToe) -> int:
        """
        Runs the MCTS algorithm for a fixed number of simulations and returns the
        best move based on the most visited child node.
        """
        if game.is_game_over():
            raise Exception("Cannot get move from a terminal game state.")

        # 1. Initialize the root of the MCTS tree
        root = MCTSNode(game)
        
        for _ in range(self.simulations):
            # A. Selection: Traverse down the tree using UCT until an unexpanded node
            node = self._select_node(root)
            
            # B. Expansion: Add a new child node (if not terminal)
            if not node.game_state.is_game_over():
                node = self._expand_node(node)

            # C. Simulation: Playout a random game from the new node
            score = self._simulate(node.game_state)
            
            # D. Backpropagation: Update wins/visits up the tree
            self._backpropagate(node, score)

        # 5. Final Move Decision: Choose the move that leads to the most visited child
        return self._best_move(root)

    def _select_node(self, node: MCTSNode) -> MCTSNode:
        """The Selection phase: Traverse the tree using UCT."""
        while node.is_fully_expanded() and not node.game_state.is_game_over():
            _, node = node.best_uct_child(self.c_param)
        return node

    def _expand_node(self, node: MCTSNode) -> MCTSNode:
        """The Expansion phase: Select an unexpanded move and create a new child."""
        unexpanded_moves = node.unexpanded_moves()
        move = random.choice(unexpanded_moves)
        
        new_game_state = node.game_state.make_move(move)
        new_node = MCTSNode(new_game_state, parent=node, parent_move=move)
        node.children[move] = new_node
        
        return new_node

    def _simulate(self, game: TicTacToe) -> int:
        """
        The Simulation (or Playout) phase: Play a random game until a terminal state.
        Returns the winner (1, -1, or 0 for draw).
        """
        current_game = game
        while not current_game.is_game_over():
            legal_moves = current_game.get_legal_moves()
            if not legal_moves: # Should be handled by is_game_over but good for safety
                return 0 
            move = random.choice(legal_moves)
            current_game = current_game.make_move(move)

        # winner  = typing.cast(int, current_game.check_win())

        return typing.cast(int, current_game.check_win())   

    def _backpropagate(self, node: MCTSNode, result: int) -> None:
        """The Backpropagation phase: Update visits and wins up to the root."""
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            
            # Score is from the perspective of the player *who just played* to reach the current_node's state
            # This player is current_node.game_state.current_player * -1
            
            # The result is the final winner (1, -1, or 0)
            if result == 0:
                score = 0.5 # Draw
            elif result == current_node.game_state.current_player * -1:
                score = 1.0 # Win
            else:
                score = 0.0 # Loss

            current_node.wins += score
            current_node = current_node.parent

    def _best_move(self, root: MCTSNode) -> int:
        """
        The final decision: Choose the move corresponding to the child with the most visits.
        """
        # We look for the most visited child, which is often more stable than the one with the highest win rate.
        best_visits = -1
        best_move = -1
        
        for move, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
                
        if best_move == -1:
            raise Exception("MCTS failed to find a move for the current state.")
            
        return best_move