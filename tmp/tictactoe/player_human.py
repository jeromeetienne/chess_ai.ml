import sys

from .tictactoe import TicTacToe
from .player_protocol import PlayerProtocol


class HumanPlayer:
    """
    Represents a player whose moves are decided by human input.
    """
    def __init__(self, player_id: int):
        self.player_id: int = player_id
        self.marker: str = 'X' if player_id == 1 else 'O'

        assert isinstance(self, PlayerProtocol), "HumanPlayer does not conform to Player protocol."

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
