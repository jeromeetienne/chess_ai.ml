# stdlib imports
import os

# pip imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# local imports
from src.libs.early_stopper import EarlyStopper
from src.utils.pgn_utils import PGNUtils


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.abspath(os.path.join(__dirname__, '..', 'data'))
tensors_folder_path = os.path.join(data_folder_path, 'pgn_tensors')
# Define the Model
################################################################
class RegressionModel(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for regression.
    It processes the (21, 8, 8) input and outputs a single float.
    """
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Input: (batch_size, 21, 8, 8) -> 21 is the number of channels

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv1: Input 21 channels, Output 16 channels. Kernel size 3x3.
            nn.Conv2d(in_channels=21, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),

        )

        # Calculate the size of the flattened tensor after conv layers
        # Input is (128 channels * 2 * 2 = 128 features)
        self.flattened_size = 128 * 4*4  # C * H * W

        # Fully connected layers (for regression output)
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output a single float
        )

    def forward(self, x):
        x = x.to(torch.float32)
        # x shape: (batch_size, 21, 8, 8)
        x = self.conv_layers(x)
        # x shape: (batch_size, 32, 2, 2)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        # x shape: (batch_size, 128)
        x = self.fc_layers(x)
        # x shape: (batch_size, 1)
        return x

# ----------------------------------------------------------------

# 1. Setup Data and Hyperparameters
################################################################
# Check for GPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 300
DATA_SIZE = 10000

from src.utils.dataset_utils import DatasetUtils

# torch.manual_seed(42)

pgn_paths = PGNUtils.get_pgn_paths()
# truncate file_pgn_paths to max_files_count
pgn_paths = pgn_paths[:3]

basename_prefix = '5b07b25e0ebc5914abc12c6d.split_01_of_66'
boards_tensor, moves_tensor = DatasetUtils.load_dataset_tensor(tensors_folder_path, f'{basename_prefix}')

eval_path = os.path.join(tensors_folder_path, f'{basename_prefix}{DatasetUtils.FILE_SUFFIX.EVALS}')
evals_tensor = torch.load(eval_path)

evals_tensor = evals_tensor.view(-1, 1)  # Reshape to (N, 1)

# downsample for quicker testing
boards_tensor = boards_tensor[:DATA_SIZE]
moves_tensor = moves_tensor[:DATA_SIZE]
evals_tensor = evals_tensor[:DATA_SIZE]

print(DatasetUtils.dataset_summary(boards_tensor, moves_tensor))
print(f'Evals tensor shape: {evals_tensor.shape}, dtype: {evals_tensor.dtype}')

# Create Dataset and DataLoader
dataset = TensorDataset(boards_tensor, evals_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------------------------------------------

# 2. Instantiate Model, Loss, and Optimizer
################################################################
model = RegressionModel().to(device)
criterion = nn.MSELoss()  # Mean Squared Error is common for regression
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------------------------------------------
# Add a learning rate scheduler to reduce LR over time
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, threshold=0.001)

early_stopper = EarlyStopper(patience=30, threshold=0.00001)

# 3. Training Loop
################################################################
print(f"Starting training on {device}...")
model.train()  # Set the model to training mode

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for batch_index, (board_inputs,eval_outputs) in enumerate(dataloader):
        # Get inputs and labels, and move them to the device
        board_inputs = board_inputs.to(device)
        eval_outputs = eval_outputs.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(board_inputs)

        # Calculate loss
        loss = criterion(outputs, eval_outputs)

        # Backward pass (calculate gradients)
        loss.backward()

        # Update weights
        optimizer.step()

        # Track loss
        running_loss += loss.item()
    
    # Print statistics every epoch
    training_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch+1} lr={scheduler.get_last_lr()[0]} Loss: {training_loss:.8f}')

    # Step the scheduler
    scheduler.step(training_loss)

    # Check for early stopping
    must_stop, must_save = early_stopper.early_stop(training_loss)

    if must_stop:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break


print("Training finished!")
# 

# ----------------------------------------------------------------

# 4. Example Prediction (Optional)
################################################################
# Switch to evaluation mode
model.eval()

# Create a single dummy input (batch size of 1)

for i in range(40):
    test_input = boards_tensor[i:i+1].to(device)  # Use an actual sample from the dataset

    # Disable gradient calculation for inference
    with torch.no_grad():
        prediction = model(test_input)

    print(f"Example Prediction for a single input: {prediction.item():-3.4f} Actual eval: {evals_tensor[i].item():4.4f} delta={abs(prediction.item() - evals_tensor[i].item()):.4f}")