import math
from typing import Dict, List, Optional, Tuple
# Import existing classes from the previous code

from .tictactoe import TicTacToe


class MCTSNode:
    """
    Represents a single node in the Monte Carlo Tree Search tree.
    """
    def __init__(self, game_state: TicTacToe, parent: Optional['MCTSNode'] = None, parent_move: Optional[int] = None):
        self.game_state: TicTacToe = game_state
        self.parent: Optional['MCTSNode'] = parent
        self.parent_move: Optional[int] = parent_move # The move that led to this state
        self.children: Dict[int, 'MCTSNode'] = {}    # Maps move (int) to child node
        self.wins: float = 0.0                      # Total wins from this node's perspective (1 for win, 0.5 for draw, 0 for loss)
        self.visits: int = 0                        # Total number of times this node has been visited
    
    def is_fully_expanded(self) -> bool:
        """Checks if all legal moves from this state have corresponding child nodes."""
        return len(self.children) == len(self.game_state.get_legal_moves())

    def best_uct_child(self, c_param: float = 1.4) -> Tuple[int, 'MCTSNode']:
        """
        Selects the child node with the highest UCT1 (Upper Confidence Bound 1 applied to trees) value.
        UCT1 formula: (wins / visits) + c * sqrt(ln(parent_visits) / visits)
        """
        log_parent_visits = math.log(self.visits)
        
        # We want to maximize the UCT score
        best_score = -float('inf')
        best_move_node: Optional[Tuple[int, MCTSNode]] = None

        for move, child in self.children.items():
            if child.visits == 0:
                # Prioritize unvisited nodes for expansion
                score = float('inf') 
            else:
                # The win rate is from the perspective of the player *who just played* to reach this node.
                # When we are selecting, we are choosing the move for the *current* player (game_state.current_player).
                
                # UCT calculation: win_rate + exploration_term
                win_rate = child.wins / child.visits
                exploration_term = c_param * math.sqrt(log_parent_visits / child.visits)
                score = win_rate + exploration_term
            
            if score > best_score:
                best_score = score
                best_move_node = (move, child)
        
        if best_move_node is None:
             raise Exception("No children found for UCT selection, this should not happen in a non-terminal node.")

        return best_move_node

    def unexpanded_moves(self) -> List[int]:
        """Returns a list of legal moves that do not yet have a child node."""
        all_moves = set(self.game_state.get_legal_moves())
        expanded_moves = set(self.children.keys())
        return list(all_moves - expanded_moves)