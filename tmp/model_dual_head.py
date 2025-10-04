# stdlib imports
import os

# pip imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import tqdm

# local imports
from src.libs.early_stopper import EarlyStopper
from src.utils.pgn_utils import PGNUtils


__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.abspath(os.path.join(__dirname__, "..", "output"))
tensors_folder_path = os.path.join(output_folder_path, "pgn_tensors")


# Define the Model
################################################################
class DualHeadModel(nn.Module):
    """
    Convolutional network with two heads:
    - policy head: classification logits over CLASS_COUNT classes
    - value head: regression scalar (single float)
    Input: (batch, 21, 8, 8)
    """

    def __init__(self, class_count: int = 1968):
        super(DualHeadModel, self).__init__()
        self.class_count = class_count

        # Shared convolutional trunk
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=21, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Flattened size: channels * H * W
        self.flattened_size = 128 * 8 * 8

        # Shared fully connected layer
        self.shared_fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
        )

        # Policy (classification) head
        self.policy_head = nn.Linear(256, self.class_count)

        # Value (regression) head
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.shared_fc(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


# ----------------------------------------------------------------

# 1. Setup Data and Hyperparameters
################################################################
# Check for GPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore

# Hyperparameters
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
NUM_EPOCHS = 300
DATA_SIZE = 10000

from src.utils.dataset_utils import DatasetUtils

torch.manual_seed(42)

# Load dataset (boards, moves, evals)
boards_tensor, moves_tensor, evals_tensor = DatasetUtils.load_datasets(tensors_folder_path, max_file_count=10)

# Ensure shapes/dtypes: moves -> long (class indices), evals -> float, shape (N,1)
# if moves_tensor.dim() > 1:
#     moves_tensor = moves_tensor.view(-1)
# moves_tensor = moves_tensor.to(torch.long)

evals_tensor = evals_tensor.view(-1, 1).to(torch.float32)

# downsample for quicker testing
boards_tensor = boards_tensor[:DATA_SIZE]
moves_tensor = moves_tensor[:DATA_SIZE]
evals_tensor = evals_tensor[:DATA_SIZE]

print(DatasetUtils.dataset_summary(boards_tensor, moves_tensor, evals_tensor))
print(f"Evals tensor shape: {evals_tensor.shape}, dtype: {evals_tensor.dtype}")

# Create Dataset and DataLoader (boards, moves, evals)
dataset = TensorDataset(boards_tensor, moves_tensor, evals_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------------------------------------------

# 2. Instantiate Model, Loss, and Optimizer
################################################################
CLASS_COUNT = 1968

# Instantiate model, losses and optimizer
model = DualHeadModel(class_count=CLASS_COUNT).to(device)
policy_criterion = nn.CrossEntropyLoss()
value_criterion = nn.MSELoss()

# Weighting for combined loss (policy_loss * POLICY_WEIGHT + value_loss * VALUE_WEIGHT)
POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 1.0

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------------------------------------------
# Add a learning rate scheduler to reduce LR over time
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, threshold=0.001)

early_stopper = EarlyStopper(patience=30, threshold=0.00001)

# 3. Training Loop
################################################################
print(f"Starting training on {device}...")
model.train()

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    running_policy_loss = 0.0
    running_value_loss = 0.0
    running_accuracy = 0.0
    for batch_index, (board_inputs, move_targets, eval_outputs) in enumerate(tqdm.tqdm(dataloader, ncols=80)):
        board_inputs = board_inputs.to(device)
        move_targets = move_targets.to(device)
        eval_outputs = eval_outputs.to(device)

        optimizer.zero_grad()

        policy_logits, value_preds = model(board_inputs)

        # policy_logits: (batch, CLASS_COUNT)
        # move_targets: (batch,) long
        policy_loss = policy_criterion(policy_logits, move_targets)

        # value_preds: (batch,1), eval_outputs: (batch,1)
        value_loss = value_criterion(value_preds, eval_outputs)

        loss = POLICY_WEIGHT * policy_loss + VALUE_WEIGHT * value_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_policy_loss += policy_loss.item()
        running_value_loss += value_loss.item()

        # compute accuracy
        with torch.no_grad():
            predictions = torch.argmax(policy_logits, dim=1)
            batch_accuracy = (predictions == move_targets).float().mean().item()
            running_accuracy += batch_accuracy

    training_loss = running_loss / len(dataloader)
    avg_policy_loss = running_policy_loss / len(dataloader)
    avg_value_loss = running_value_loss / len(dataloader)
    average_accuracy = running_accuracy / len(dataloader)

    print(
        f"Epoch {epoch+1} learning_rate={scheduler.get_last_lr()[0]} Loss: {training_loss:.6f} (policy={avg_policy_loss:.6f}, value={avg_value_loss:.6f}) accuracy={average_accuracy:.4f}"
    )

    scheduler.step(training_loss)

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
for i in range(min(40, len(boards_tensor))):
    test_input = boards_tensor[i : i + 1].to(device)
    true_move = moves_tensor[i].item()
    true_eval = evals_tensor[i].item()
    with torch.no_grad():
        policy_logits, value_pred = model(test_input)
        pred_move = torch.argmax(policy_logits, dim=1).item()
        pred_eval = value_pred.item()

    print(f"Sample {i}: pred_move={pred_move} true_move={true_move} pred_eval={pred_eval: .4f} true_eval={true_eval: .4f} delta={abs(pred_eval-true_eval):.4f}")
