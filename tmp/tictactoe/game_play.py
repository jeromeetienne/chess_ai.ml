from typing import List, Optional
from tmp.tictactoe.tictactoe import TicTacToe
from tmp.tictactoe.player_human import HumanPlayer
from tmp.tictactoe.player_random import RandomPlayer
from tmp.tictactoe.mtcs_player import MCTSPlayer
from tmp.tictactoe.player_protocol import PlayerProtocol

def play_tictactoe_game(human_starts: bool = True) -> None:
    """
    Plays a game of Tic-Tac-Toe between a HumanPlayer and a RandomPlayer.

    :param human_starts: If True, HumanPlayer is 'X' (1); otherwise, RandomPlayer is 'X'.
    """
    game: TicTacToe = TicTacToe()

    if human_starts:
        player1 = HumanPlayer(1) # X
        # player2 = RandomPlayer(-1) # O
        player2 = MCTSPlayer(-1) # O
    else:
        # player1 = RandomPlayer(1) # X
        player1 = MCTSPlayer(1) # X
        player2 = HumanPlayer(-1) # O

    players: dict[int, PlayerProtocol] = {1: player1, -1: player2}

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
    choice = input("Who do you want to play as? Enter 'X' to play first, or 'O' to play second: ").strip().upper()
    if choice == 'X':
        print("\n--- Starting a new game (Human is 'X', Random AI is 'O') ---")
        play_tictactoe_game(human_starts=True)
    elif choice == 'O':
        print("\n--- Starting a new game (Random AI is 'X', Human is 'O') ---")
        play_tictactoe_game(human_starts=False)
    else:
        print("Invalid choice. Please run the program again and enter 'X' or 'O'.")