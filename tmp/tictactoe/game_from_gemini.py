# https://gemini.google.com/app/2bb50ea38c81aa30
import random
import sys
from typing import List, Optional, Protocol, runtime_checkable

# --- TicTacToe Class (Unchanged) ---
class TicTacToe:
    """
    Represents the state and rules of a Tic-Tac-Toe game.
    """
    def __init__(self, size: int = 3) -> None:
        self.size: int = size
        # The board is a flattened list for easy representation,
        # where 0=Empty, 1='X' (Player 1), -1='O' (Player -1)
        self.board: List[int] = [0] * (size * size)
        # 1: 'X', -1: 'O'
        self.current_player: int = 1

    def __repr__(self) -> str:
        """Prints a human-readable board representation, showing move indices on empty cells."""
        s: str = ""
        for i in range(self.size):
            row = self.board[i * self.size : (i + 1) * self.size]
            s += " | ".join(["X" if cell == 1 else "O" if cell == -1 else str(i * self.size + j) for j, cell in enumerate(row)])
            s += "\n"
            if i < self.size - 1:
                s += "--" + "+---" * (self.size - 1) + "\n"
        return s

    def get_legal_moves(self) -> List[int]:
        """Returns a list of indices (0-8) where moves can be made."""
        return [i for i, cell in enumerate(self.board) if cell == 0]
    
    def get_game_state(self) -> List[int]:
        """Returns the current game state as a list."""
        return list(self.board)

    def make_move(self, move: int) -> "TicTacToe":
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

    def check_win(self) -> Optional[int]:
        """
        Checks for a win. Returns 1 if 'X' wins, -1 if 'O' wins, 0 if no winner,
        and None if the game is still ongoing.
        """
        
        # Check rows, columns, and diagonals
        lines: List[List[int]] = []
        # Rows
        for i in range(self.size):
            lines.append(self.board[i * self.size : (i + 1) * self.size])
        # Columns
        for i in range(self.size):
            lines.append(self.board[i::self.size])
        # Diagonals (Specific logic for 3x3, but robust for the standard case)
        diag1 = self.board[::self.size + 1]
        diag2 = self.board[self.size - 1: self.size * self.size - 1: self.size - 1]
            
        lines.append(diag1)
        lines.append(diag2)
        
        for line in lines:
            if sum(line) == self.size:
                return 1  # 'X' wins
            if sum(line) == -self.size:
                return -1 # 'O' wins

        # Check for draw (if no moves left)
        if not self.get_legal_moves():
            return 0  # Draw

        return None # Game is still ongoing

    def is_game_over(self) -> bool:
        """True if there's a winner or a draw."""
        return self.check_win() is not None

# ------------------------------
# --- Player Classes (New) ---
# ------------------------------

@runtime_checkable
class Player(Protocol):
    """
    Protocol/Interface for any TicTacToe player (Human or AI).
    A player must have a player_id (1 or -1) and a get_move method.
    """
    player_id: int
    marker: str

    def get_move(self, game: TicTacToe) -> int:
        """
        Calculates and returns the player's chosen move (index 0-8).
        """
        ...

class HumanPlayer:
    """
    Represents a player whose moves are decided by human input.
    """
    def __init__(self, player_id: int):
        self.player_id: int = player_id
        self.marker: str = 'X' if player_id == 1 else 'O'

    def get_move(self, game: TicTacToe) -> int:
        """
        Prompts the human for a move and validates the input.
        """
        print(f"👤 Your Turn ({self.marker}). Legal moves are: {game.get_legal_moves()}")
        while True:
            try:
                move_input = input("Enter your move (index 0-8): ")
                # Check for EOF/Ctrl+D and exit gracefully
                if move_input is None:
                    raise EOFError()
                
                move = int(move_input)
                legal_moves = game.get_legal_moves()
                
                if move in legal_moves:
                    return move
                else:
                    print(f"❌ Invalid move: {move} is not an empty cell or is outside the board range.")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
            except EOFError:
                print("\nGame aborted by user.")
                sys.exit(0) # Exit the program cleanly

class RandomPlayer:
    """
    Represents an AI player that chooses a move randomly from legal options.
    """
    def __init__(self, player_id: int):
        self.player_id: int = player_id
        self.marker: str = 'X' if player_id == 1 else 'O'

    def get_move(self, game: TicTacToe) -> int:
        """
        Picks a random move from the list of legal moves.
        """
        print(f"🤖 AI's Turn ({self.marker}). Thinking...")
        legal_moves = game.get_legal_moves()
        if legal_moves:
            ai_move = random.choice(legal_moves)
            print(f"AI chooses move: {ai_move}")
            return ai_move
        
        # This should ideally not be reached if is_game_over is checked first
        raise Exception("AI attempted to move when no legal moves were available (Draw state).")


# ---------------------------------------------
# --- Game Playing Function (Updated) ---
# ---------------------------------------------

def play_tictactoe_game(human_starts: bool = True) -> None:
    """
    Plays a game of Tic-Tac-Toe between a HumanPlayer and a RandomPlayer.

    :param human_starts: If True, HumanPlayer is 'X' (1); otherwise, RandomPlayer is 'X'.
    """
    game: TicTacToe = TicTacToe()

    if human_starts:
        player1 = HumanPlayer(1) # X
        player2 = RandomPlayer(-1) # O
    else:
        player1 = RandomPlayer(1) # X
        player2 = HumanPlayer(-1) # O

    players: dict[int, Player] = {1: player1, -1: player2}

    # Display the initial board with move indices
    print("🤖 Welcome to Tic-Tac-Toe! The board indices are as follows:")
    print(game) # The __repr__ now includes indices on empty cells
    
    print("-" * 30)
    print(f"Player X is: {type(players[1]).__name__}")
    print(f"Player O is: {type(players[-1]).__name__}")
    print("-" * 30)
    
    while not game.is_game_over():
        current_player_obj = players[game.current_player]
        
        print(f"\n✨ Current Board:\n{game}")
        
        # Get the move from the current player object
        move = current_player_obj.get_move(game)
        
        # Make the move and update the game state
        game = game.make_move(move)
            
    # Game Over
    print("\n--- Game Over ---")
    print(game)
    result = game.check_win()
    
    if result == 1:
        print(f"🎉 **Player X ({players[1].marker}) Wins!** 🎉")
    elif result == -1:
        print(f"💔 **Player O ({players[-1].marker}) Wins!** 💔")
    else: # result == 0
        print("🤝 **It's a Draw!** 🤝")

# --- Example of How to Play ---
if __name__ == "__main__":
    # Human starts as 'X'
    print("\n--- Starting a new game (Human is 'X', Random AI is 'O') ---")
    play_tictactoe_game(human_starts=True)

    # # Uncomment to play a game where the AI starts as 'X' and human is 'O'
    # print("\n--- Starting a new game (Random AI is 'X', Human is 'O') ---")
    # play_tictactoe_game(human_starts=False)