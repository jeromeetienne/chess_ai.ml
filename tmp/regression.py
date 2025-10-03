# stdlib imports
import os

# pip imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# local imports
from src.libs.early_stopper import EarlyStopper


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
            nn.Conv2d(in_channels=21, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # # Max Pool: (8, 8) -> (4, 4)
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv2: Input 16 channels, Output 32 channels. Kernel size 3x3.
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # # Max Pool: (4, 4) -> (2, 2)
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate the size of the flattened tensor after conv layers
        # Input is (32 channels * 2 * 2 = 128 features)
        self.flattened_size = 32 * 8*8  # C * H * W

        # Fully connected layers (for regression output)
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a single float
        )

    def forward(self, x):
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
INPUT_SHAPE = (21, 8, 8)
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
NUM_EPOCHS = 1000
DATA_SIZE = 10_000

from src.utils.dataset_utils import DatasetUtils

torch.manual_seed(42)

basename_prefix = '5b07b25e0ebc5914abc12c6d.split_01_of_66'
boards_tensor, moves_tensor = DatasetUtils.load_dataset_tensor(tensors_folder_path, f'{basename_prefix}')

eval_path = os.path.join(tensors_folder_path, f'{basename_prefix}{DatasetUtils.FILE_SUFFIX.EVALS}')
evals_tensor = torch.load(eval_path)

# downsample for quicker testing
# boards_tensor = boards_tensor[:DATA_SIZE]
# moves_tensor = moves_tensor[:DATA_SIZE]
# evals_tensor = evals_tensor[:DATA_SIZE]

print(DatasetUtils.dataset_summary(boards_tensor, moves_tensor))
print(f'Evals tensor shape: {evals_tensor.shape}, dtype: {evals_tensor.dtype}')

# 1.1 Create dummy data
# Input: (DATA_SIZE, 21, 8, 8)
# Output: (DATA_SIZE, 1)
boards_tensor = torch.randn(DATA_SIZE, *INPUT_SHAPE, dtype=torch.float32)
evals_tensor = torch.randn(DATA_SIZE, 1, dtype=torch.float32)

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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, threshold=0.05)

early_stopper = EarlyStopper(patience=30, threshold=0.0001)

# 3. Training Loop
################################################################
print(f"Starting training on {device}...")
model.train()  # Set the model to training mode

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for batch_index, (inputs,labels) in enumerate(dataloader):
        # Get inputs and labels, and move them to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass (calculate gradients)
        loss.backward()

        # Update weights
        optimizer.step()

        # Track loss
        running_loss += loss.item()
    
    # Print statistics every epoch
    training_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {training_loss:.6f}')

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
test_input = torch.randn(1, *INPUT_SHAPE, dtype=torch.float32).to(device)
test_input = boards_tensor[0:1].to(device)  # Use an actual sample from the dataset

# Disable gradient calculation for inference
with torch.no_grad():
    prediction = model(test_input)

print(f"\nExample Prediction for a single input: {prediction.item():.4f}")