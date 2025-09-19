# # Chess Engine with PyTorch

# ## Imports

#
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chess import pgn
import pickle

import os
__dirname__ = os.path.dirname(os.path.abspath(__file__))

from libs.dataset import ChessDataset
from libs.model import ChessModel
from libs.utils import create_input_for_nn, encode_moves, load_games_from_pgn


###############################################################################
# Load PGN files and parse games
#
pgn_folder_path = f"{__dirname__}/../data/pgn"
pgn_file_paths = [file for file in os.listdir(pgn_folder_path) if file.endswith(".pgn")]
# sort files alphabetically to ensure consistent order
pgn_file_paths.sort(reverse=False)

# truncate file_pgn_paths to max_files_count
max_files_count = 28
# max_files_count = 18
max_files_count = 10
pgn_file_paths = pgn_file_paths[:max_files_count]

games: list[pgn.Game] = []
for file_index, pgn_file_path in enumerate(pgn_file_paths):
    print(f'processing file {pgn_file_path} ({file_index+1}/{len(pgn_file_paths)})')
    new_games = load_games_from_pgn(f"{pgn_folder_path}/{pgn_file_path}")
    games.extend(new_games)

print(f"GAMES LOADED: {len(games)}")

# Shuffle the games
# random_seed = 42
# torch.manual_seed(random_seed)
# games_rnd_indexes = torch.randperm(len(games)).tolist()
# games = [games[i] for i in games_rnd_indexes]

# keep only max_games_count games
max_games_count = len(games)
# max_games_count = 2_000
# max_games_count = 1_000
max_games_count = 500
games = games[:max_games_count]
#
print(f"GAMES PARSED: {len(games)}")


###############################################################################
# Convert data into tensors
#
X, y = create_input_for_nn(games)

print(f"NUMBER OF SAMPLES: {len(y)}")

# Truncate to 2.5 million samples
X = X[0:2500000]
y = y[0:2500000]

# Encode moves
y, move_to_int = encode_moves(y)
num_classes = len(move_to_int)
print(f"NUMBER OF UNIQUE MOVES: {num_classes}")


# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

###############################################################################
# Preliminary actions
#

# Create Dataset and DataLoader
dataset = ChessDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Check for GPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f'Using device: {device}')

# Model Initialization
model = ChessModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(chess_model.parameters(), lr=0.001, weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

###############################################################################
# Display model summary

print(model)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:_}")

###############################################################################
# Training
#

num_epochs = 20
num_epochs = 2
for epoch in range(num_epochs):
    time_start = time.time()
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()

        outputs = model(inputs)  # Raw logits

        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()
    
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()
    time_end = time.time()
    time_elapsed = time_end - time_start
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}, Time: {time_elapsed:.2f}-sec')

###############################################################################
# Save the model
#

# Save the model
state_dict_path = f"{__dirname__}/../output/model.pth"
torch.save(model.state_dict(), state_dict_path)


# save move_to_int mapping
move_to_int_path = f"{__dirname__}/../output/move_to_int.pickle"
with open(move_to_int_path, "wb") as pgn_file_path:
    pickle.dump(move_to_int, pgn_file_path)

# Write a README file with training details
readme_md = f"""# Chess Model Training
- Model trained on {len(games)} games from {len(pgn_file_paths)} PGN files.
- Number of unique moves: {num_classes}
- Number of samples: {len(y)}
- Number of epochs: {num_epochs}
- Final Loss: {running_loss / len(dataloader):.4f}
- model: {model}
"""

README_path = f"{__dirname__}/../output/README.md"
with open(README_path, "w") as readme_file:
    readme_file.write(readme_md)
