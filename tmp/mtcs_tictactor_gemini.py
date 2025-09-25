# from https://gemini.google.com/app/aac06299ea863cae
import math
import random

from tmp.mcts_tictactoe_chatgpt import mcts

# --- TicTacToe Game State Class ---
class TicTacToe:
    """
    Represents the state and rules of a Tic-Tac-Toe game.
    """
    def __init__(self, size=3):
        self.size = size
        # The board is a flattened list for easy representation,
        # where 0=Empty, 1='X' (Player 1), -1='O' (Player -1)
        self.board = [0] * (size * size)
        # 1: 'X', -1: 'O'
        self.current_player = 1

    def __repr__(self):
        """Prints a human-readable board representation."""
        s = ""
        for i in range(self.size):
            row = self.board[i * self.size : (i + 1) * self.size]
            s += " | ".join(["X" if cell == 1 else "O" if cell == -1 else " " for cell in row])
            s += "\n"
            if i < self.size - 1:
                s += "---" + "+---" * (self.size - 1) + "\n"
        return s

    def get_legal_moves(self):
        """Returns a list of indices (0-8) where moves can be made."""
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def make_move(self, move):
        """
        Creates and returns a new TicTacToe object after making the move.
        Assumes the move is valid.
        """
        if self.board[move] != 0:
            raise ValueError("Invalid move attempted on a non-empty cell.")
            
        new_game = TicTacToe(self.size)
        new_game.board = list(self.board)  # Deep copy the board
        new_game.board[move] = self.current_player
        new_game.current_player = -self.current_player  # Switch player
        return new_game

    def check_win(self):
        """
        Checks for a win. Returns 1 if 'X' wins, -1 if 'O' wins, 0 if no winner,
        and None if the game is still ongoing.
        """
        
        # Check rows, columns, and diagonals
        lines = []
        # Rows
        for i in range(self.size):
            lines.append(self.board[i * self.size : (i + 1) * self.size])
        # Columns
        for i in range(self.size):
            lines.append(self.board[i::self.size])
        # Diagonals
        lines.append(self.board[::self.size + 1])  # Main diagonal
        lines.append(self.board[self.size - 1: self.size * self.size - 1: self.size - 1]) # Anti-diagonal
        
        for line in lines:
            if sum(line) == self.size:
                return 1  # 'X' wins
            if sum(line) == -self.size:
                return -1 # 'O' wins

        # Check for draw (if no moves left)
        if not self.get_legal_moves():
            return 0  # Draw

        return None # Game is still ongoing

    def is_game_over(self):
        """True if there's a winner or a draw."""
        return self.check_win() is not None

# --- MCTS Node Class ---
class MCTSNode:
    """
    Represents a node in the MCTS tree.
    """
    def __init__(self, state, parent=None, move=None):
        self.state = state          # The TicTacToe state
        self.parent = parent        # Parent MCTSNode
        self.move = move            # The move that led to this state
        self.wins = 0               # Number of wins from this node's simulations
        self.visits = 0             # Total number of visits to this node
        self.children = {}          # Dictionary of {move: MCTSNode}
        self.unexplored_moves = state.get_legal_moves() # List of moves yet to be explored

    def is_fully_expanded(self):
        """True if all legal moves have a corresponding child node."""
        return not self.unexplored_moves and self.children

    def best_uct_child(self, C_param=1.4):
        """
        Selects the child with the highest UCT score.
        UCT (Upper Confidence Bound 1 applied to Trees) is the selection criterion.
        """
        choices_weights = []
        for child in self.children.values():
            # UCT Formula: Q + C * sqrt( (2 * ln(N)) / n )
            # Q = wins/visits (Exploitation)
            # N = parent.visits
            # n = child.visits
            
            # The win value for the child is from the perspective of the player *to move* # in the child's state, which is -state.current_player from the parent.
            # We treat the 'wins' as value for the parent's player, so for the child, 
            # we need to consider whose turn it is in the *parent* (self.state.current_player).
            # We are interested in the win rate of the child *from its own perspective* # for the player who just moved to get there.
            
            # Simplified approach: Use child.wins / child.visits as the average reward.
            # MCTS is usually implemented with rewards relative to the *just moved* player.
            # In TicTacToe, rewards are absolute (1, 0, -1). 
            # We need to account for whose turn it is.
            # If the current player is 'X' (1), we want high wins.
            # If the current player is 'O' (-1), we want high *negative* wins.
            
            # Correct UCT for two-player zero-sum:
            # Value is relative to the player *to play* in the current node.
            # The player to play in the child node is -self.state.current_player.
            
            # The Q (exploitation) term:
            # We want the Q-value from the perspective of the *current* player (self.state.current_player).
            # The Q-value stored in the child is from the perspective of the *child's* player (child.state.current_player, which is -self.state.current_player).
            # To get the value for the *current* player, we use - (child.wins / child.visits) because it's a zero-sum game.
            
            if child.visits == 0:
                # Should not happen in a typical MCTS, but as a safeguard
                # Assign a very high value to prioritize unvisited nodes
                exploit = float('inf')
            else:
                exploit = (child.wins / child.visits) * child.state.current_player # Value from child's perspective
                exploit = -exploit # Value from parent's perspective (zero-sum)

            # The exploration term:
            explore = C_param * math.sqrt(math.log(self.visits) / child.visits)
            
            uct_score = exploit + explore
            choices_weights.append((uct_score, child))

        return max(choices_weights, key=lambda x: x[0])[1]

# --- MCTS Algorithm Implementation ---
class MCTS:
    """
    Implements the MCTS search process.
    """
    def __init__(self, game_state, iterations=1000):
        self.root = MCTSNode(game_state)
        self.iterations = iterations

    def run_mcts(self):
        """Runs the main MCTS loop for a fixed number of iterations."""
        for _ in range(self.iterations):
            node = self._select(self.root)
            if node.state.is_game_over():
                # Game over: just backpropagate the final result
                result = node.state.check_win()
                self._backpropagate(node, result)
                continue
                
            # Expansion Phase
            if not node.is_fully_expanded():
                node = self._expand(node)
            
            # Simulation Phase
            result = self._simulate(node.state)
            
            # Backpropagation Phase
            self._backpropagate(node, result)
            
        # Select the best move from the root's children (most visited)
        best_child = max(self.root.children.values(), key=lambda child: child.visits)

        return best_child.move

    def _select(self, node):
        """Selection phase: Traverse the tree using UCT until an unexpanded or leaf node is found."""
        while node.is_fully_expanded() and not node.state.is_game_over():
            node = node.best_uct_child()
        return node

    def _expand(self, node):
        """Expansion phase: Create one new child node from an unexplored move."""
        move = node.unexplored_moves.pop()
        new_state = node.state.make_move(move)
        child_node = MCTSNode(new_state, parent=node, move=move)
        node.children[move] = child_node
        return child_node

    def _simulate(self, state):
        """
        Simulation phase: Play a random game from the current state until the end.
        Returns the result: 1 (for X win), -1 (for O win), 0 (for draw).
        """
        current_state = state
        while not current_state.is_game_over():
            moves = current_state.get_legal_moves()
            move = random.choice(moves)
            current_state = current_state.make_move(move)
            
        return current_state.check_win()

    def _backpropagate(self, node, result):
        """
        Backpropagation phase: Update the wins and visits from the leaf node up to the root.
        The result is the final score (1, 0, or -1).
        The win count is adjusted for the player who *just moved* to the node.
        """
        while node is not None:
            node.visits += 1
            # Add the result if the player who *just moved* (which is the opponent 
            # of the current player in 'node.state') won.
            # result is 1 (X wins) or -1 (O wins).
            # node.move is the move that led to 'node'. The player who made 
            # node.move is -node.state.current_player.
            
            # We want to know: Did the player who *just moved* to this node win?
            # The player who just moved is the one whose turn it *was* in the parent,
            # which is the player **-node.state.current_player**.
            # If the result matches the player who just moved, it's a win for them.
            player_who_moved = -node.state.current_player if node.state.current_player is not None else 1
            
            # The 'wins' count is relative to the player who *just moved* to the node.
            # If the game result matches the player who moved, they get a 'win' (1).
            if result == player_who_moved:
                node.wins += 1
            # If the result is a draw (0), it contributes 0.
            # If the result is a loss (result == -player_who_moved), it contributes 0. 
            # In some MCTS implementations, the loss is -1, but for Tic-Tac-Toe, 
            # simply adding 1 for a win and 0 for a loss/draw works fine for the 
            # exploitation term's ratio (wins/visits).
            
            node = node.parent

# --- Game Loop/Example Usage ---
def play_mcts_game():
    """Runs an interactive game of Tic-Tac-Toe against the MCTS AI."""
    print("--- Tic-Tac-Toe MCTS AI ---")
    game = TicTacToe()
    
    # Choose who goes first
    first_player = input("Do you want to go first (1) or second (2)? ").strip()
    if first_player == '1':
        human_player = 1 # 'X'
        ai_player = -1   # 'O'
    elif first_player == '2':
        human_player = -1 # 'O'
        ai_player = 1    # 'X'
    else:
        print("Invalid choice, defaulting to going first (You are 'X').")
        human_player = 1
        ai_player = -1

    print(f"\nYour marker: {'X' if human_player == 1 else 'O'}")
    print(f"AI marker: {'X' if ai_player == 1 else 'O'}")
    print("Board layout indices:\n0 | 1 | 2\n--+---+--\n3 | 4 | 5\n--+---+--\n6 | 7 | 8\n")
    print(game)


    while not game.is_game_over():
        current_marker = 'X' if game.current_player == 1 else 'O'
        
        if game.current_player == human_player:
            # Human's Turn
            while True:
                try:
                    move = int(input(f"Your turn ({current_marker}). Enter move (0-8): "))
                    if move in game.get_legal_moves():
                        game = game.make_move(move)
                        break
                    else:
                        print("Invalid or occupied move. Try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            # AI's Turn
            print(f"AI's turn ({current_marker}). Thinking...")
            # Use 5000 iterations for a reasonable challenge/speed tradeoff
            mcts = MCTS(game, iterations=50_000) 
            best_move = mcts.run_mcts()
            
            print(f"AI chose move: {best_move}")
            game = game.make_move(best_move)
            
        print("\n" + str(game))
        
    # Game over
    winner = game.check_win()
    if winner == 1:
        print("Game Over. X wins!")
    elif winner == -1:
        print("Game Over. O wins!")
    else:
        print("Game Over. It's a draw!")

# Uncomment the line below to run the game
play_mcts_game()