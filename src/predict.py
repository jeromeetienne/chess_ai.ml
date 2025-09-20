import torch
import pickle
import numpy as np
import os
import chess
import chess.pgn as pgn
import random

from libs.utils import board_to_matrix
from libs.model import ChessModel

__dirname__ = os.path.dirname(os.path.abspath(__file__))

###############################################################################
# 1. Prepare the Input Data
#

# Convert the board state to a format suitable for the model:


#
def prepare_input(board: chess.Board) -> torch.Tensor:
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor


###############################################################################
# 2. Load the Model & mapping and Move to GPU if Available
#

#
# Load the mapping

move_to_int_path = f"{__dirname__}/../output/move_to_int.pickle"
with open(move_to_int_path, "rb") as file:
    move_to_int = pickle.load(file)

# Check for GPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device: {device}")

# Load the model
model = ChessModel(num_classes=len(move_to_int))
model_path = f"{__dirname__}/../output/model.pth"
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()  # Set the model to evaluation mode (it may be reductant)

int_to_move = {v: k for k, v in move_to_int.items()}


# Function to make predictions
# TODO move that in ./libs/utils.py
def predict_next_move(board: chess.Board) -> str | None:
    """
    Predict the best move for the given board state.
    Args:
        board (chess.Board): The current state of the chess board.
    Returns:
        str | None: The predicted best move in UCI format, or None if no legal move is found.
    """
    X_tensor = prepare_input(board).to(device)

    with torch.no_grad():
        logits = model(X_tensor)

    logits = logits.squeeze(0)  # Remove batch dimension

    probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probabilities = probabilities[sorted_indices]

    # TODO make the possibility to return top N moves with their probabilities, not just the best one
    # - thus allowing to implement some randomness in the move selection
    proposed_top_n = 5
    proposed_moves_uci_proba = []
    for move_index in sorted_indices:
        move_uci = int_to_move[move_index]
        # skip illegal moves
        if move_uci not in legal_moves_uci:
            continue

        proposed_moves_uci_proba.append((move_uci, probabilities[move_index]))
        if len(proposed_moves_uci_proba) >= proposed_top_n:  # Get top 5 legal moves
            break

    if len(proposed_moves_uci_proba) == 0:
        return None  # No legal moves found (should not happen in a normal game)

    # If it is one, it is the best move
    # if it is less than 1, it will pick randomly among the proposed moves with a probability >= best probability * random_threshold
    random_threshold = 1

    # keep only the proposed moves with a probability close to the best one
    if len(proposed_moves_uci_proba) > 1:
        best_proba = proposed_moves_uci_proba[0][1]
        threshold = random_threshold * best_proba  # Keep moves with at least 80% of the best probability
        proposed_moves_uci_proba = [(move, proba) for move, proba in proposed_moves_uci_proba if proba >= threshold]

    # sanity check
    if random_threshold == 1.0:
        assert len(proposed_moves_uci_proba) <= 1, "should be at most 1 move after thresholding"

    # return any of the proposed moves randomly
    best_move_uci = random.choice(proposed_moves_uci_proba)[0]

    return best_move_uci

    # for move_index in sorted_indices:
    #     move_uci = int_to_move[move_index]
    #     # skip illegal moves
    #     if move_uci not in legal_moves_uci:
    #         continue

    #     return move_uci  # Return the first legal move with the highest probability

    # return None


###############################################################################
# 3. Use the ```predict_move``` function to get the best move and its probabilities for a given board state:
#
# Initialize a chess board
board = chess.Board()
print(board)
print("\n")


move_count = 300
for _ in range(move_count):
    # predict the best move
    best_move = predict_next_move(board)

    if best_move is None:
        # raise an error if no legal moves are available...
        # as it a ML error, as chess module didnt declare the game over
        raise ValueError("No legal moves available. Game over.")

    turn_color = "white" if board.turn == chess.WHITE else "black"

    # show the board
    print(f"Predicted Move {board.fullmove_number} for {turn_color}: {best_move} ")

    # Make the move on the board
    board.push_uci(best_move)
    print(board)
    print("\n")

    print(f"Board Outcome: {board.outcome()}")
    if board.is_game_over():
        print("Game Over!")
        break

    # Wait for a key press to continue
    # input("Press Enter to continue...")

# Optionally, print the PGN representation of the game
print("PGN Representation of the game:")
# print(str(pgn.Game.from_board(board)))

game = pgn.Game.from_board(board)
game.headers["Event"] = "AI vs AI"
game.headers["White"] = "chess_bot.ml 1"
game.headers["Black"] = "chess_bot.ml 2"
game.headers["Result"] = board.result()

game.headers["Site"] = "My Computer"
game.headers["Round"] = "1"
# set the date to today in YYYY.MM.DD format
import datetime

today = datetime.date.today()
game.headers["Date"] = today.strftime("%Y.%m.%d")
print(game)
