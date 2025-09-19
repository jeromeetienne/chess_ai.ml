# 
import torch
import pickle
import numpy as np
import os
from chess import Board, pgn

from libs.utils import board_to_matrix
from libs.model import ChessModel

__dirname__ = os.path.dirname(os.path.abspath(__file__))


#  [markdown]
# # Predictions

#  [markdown]
# 1. Prepare the Input Data
# Convert the board state to a format suitable for the model:

# 
def prepare_input(board: Board):
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor

#  [markdown]
# 2. Load the Model & mapping and Move to GPU if Available

# 
# Load the mapping

move_to_int_path = f"{__dirname__}/../output/move_to_int.pickle"
with open(move_to_int_path, "rb") as file:
    move_to_int = pickle.load(file)

# Check for GPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f'Using device: {device}')

# Load the model
model = ChessModel(num_classes=len(move_to_int))
model_path = f"{__dirname__}/../output/model.pth"
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()  # Set the model to evaluation mode (it may be reductant)

int_to_move = {v: k for k, v in move_to_int.items()}

# Function to make predictions
def predict_move(board: Board):
    X_tensor = prepare_input(board).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
    
    logits = logits.squeeze(0)  # Remove batch dimension
    
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(probabilities)[::-1]
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return move
    
    return None

#  [markdown]
# 3. Use the ```predict_move``` function to get the best move and its probabilities for a given board state:

# 
# Initialize a chess board
board = Board()

# 
print(board)

# 
board.push_uci("e2e4")
print(board)

# 
# Predict and make a move
best_move = predict_move(board)
board.push_uci(best_move)

print(f"Predicted Move: {best_move}")
print(f"Board after move:")
print(board)

# 
print(str(pgn.Game.from_board(board)))


