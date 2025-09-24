# stdlib imports
import os

# pip imports
import torch
import chess

# local imports
from src.libs.chessbotml_player import ChessbotMLPlayer
from src.libs.io_utils import IOUtils
from src.libs.pgn_utils import PGNUtils
from src.libs.utils import Utils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = f"{__dirname__}/../../output/"

###############################################################################
# 2. Load the Model & mapping and Move to GPU if Available
#

# Load the mapping
uci_to_classindex = IOUtils.load_uci_to_classindex(folder_path=output_folder_path)
num_classes = len(uci_to_classindex)

# Load the model
model = IOUtils.load_model(folder_path=output_folder_path, num_classes=num_classes)

# Check for GPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
print(f"Using device: {device}")

# move the model to the device and set it to eval mode
model.to(device)
model.eval()  # Set the model to evaluation mode (it may be reductant)

# create the reverse mapping
classindex_to_uci: dict[int, str] = {v: k for k, v in uci_to_classindex.items()}

chatbotml_player = ChessbotMLPlayer(model=model, classindex_to_uci=classindex_to_uci)

###############################################################################
# 3. Use the ```predict_move``` function to get the best move and its probabilities for a given board state:
#

# Initialize a chess board
board = chess.Board()
print(board.unicode())
print("\n")

while True:
    # predict the best move
    best_move = chatbotml_player.predict_next_move(board)

    if best_move is None:
        # raise an error if no legal moves are available...
        # as it a ML error, as chess module didnt declare the game over
        raise ValueError("No legal moves available. Game over.")

    turn_color = "white" if board.turn == chess.WHITE else "black"

    # show the board
    print(f"Predicted Move {board.fullmove_number} for {turn_color}: {best_move} ")

    # Make the move on the board
    board.push_uci(best_move)
    print(board.unicode())
    print("\n")

    print(f"Board Outcome: {board.outcome()}")
    if board.is_game_over():
        print("Game Over!")
        break

###############################################################################
# Optionally, print the PGN representation of the game

print("PGN Representation of the game:")
pgn_game = PGNUtils.board_to_pgn(board)
print(pgn_game)
