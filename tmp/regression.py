import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
            # Max Pool: (8, 8) -> (4, 4)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv2: Input 16 channels, Output 32 channels. Kernel size 3x3.
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Max Pool: (4, 4) -> (2, 2)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate the size of the flattened tensor after conv layers
        # Input is (32 channels * 2 * 2 = 128 features)
        self.flattened_size = 32 * 2 * 2  # C * H * W

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
# print(f"Pytorch computes on {device} device")

# Hyperparameters
INPUT_SHAPE = (21, 8, 8)
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
NUM_EPOCHS = 200
DATA_SIZE = 10_000

# 1.1 Create dummy data
# Input: (DATA_SIZE, 21, 8, 8)
# Output: (DATA_SIZE, 1)
X_dummy = torch.randn(DATA_SIZE, *INPUT_SHAPE, dtype=torch.float32)
y_dummy = torch.randn(DATA_SIZE, 1, dtype=torch.float32)

# Create Dataset and DataLoader
dataset = TensorDataset(X_dummy, y_dummy)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------------------------------------------

# 2. Instantiate Model, Loss, and Optimizer
################################################################
model = RegressionModel().to(device)
criterion = nn.MSELoss()  # Mean Squared Error is common for regression
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------------------------------------------

# 3. Training Loop
################################################################
print(f"Starting training on {device}...")
model.train()  # Set the model to training mode

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        # Get inputs and labels, and move them to the device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

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
    avg_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')

print("Training finished!")
# 

# ----------------------------------------------------------------

# 4. Example Prediction (Optional)
################################################################
# Switch to evaluation mode
model.eval()

# Create a single dummy input (batch size of 1)
test_input = torch.randn(1, *INPUT_SHAPE, dtype=torch.float32).to(device)

# Disable gradient calculation for inference
with torch.no_grad():
    prediction = model(test_input)

print(f"\nExample Prediction for a single input: {prediction.item():.4f}")