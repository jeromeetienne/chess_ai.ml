import random
from typing import List, Optional
from ..protocols.game_protocol import GameProtocol


# --- TicTacToe Class (Unchanged) ---
class GameTicTacToe:
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

        # check this instance conforms to GameProtocol
        assert isinstance(self, GameProtocol), "GameTicTacToe does not conform to GameProtocol."

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

    def make_move(self, move: int) -> "GameTicTacToe":
        """
        Creates and returns a new TicTacToe object after making the move.
        Assumes the move is valid.
        """
        if self.board[move] != 0:
            raise ValueError("Invalid move attempted on a non-empty cell.")
            
        new_game = GameTicTacToe(self.size)
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

if __name__ == "__main__":
    game = GameTicTacToe()
    print("Initial Game State:")
    print(game)
    while not game.is_game_over():
        legal_moves = game.get_legal_moves()
        move = random.choice(legal_moves)
        game = game.make_move(move)
        print(f"Player {'X' if game.current_player == -1 else 'O'} made move at index {move}")
        print(game)
    
    result = game.check_win()
    if result == 1:
        print("Player X wins!")
    elif result == -1:
        print("Player O wins!")
    else:
        print("It's a draw!")