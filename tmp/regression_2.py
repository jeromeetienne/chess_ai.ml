# stdlib imports
import os
import math

# pip imports
import torch
import tqdm
import matplotlib.pyplot as plt


# local imports
from src.libs.early_stopper import EarlyStopper
from src.utils.dataset_utils import DatasetUtils


__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.abspath(os.path.join(__dirname__, "..", "output"))
tensors_folder_path = os.path.join(output_folder_path, "pgn_tensors")


# Define the Model
################################################################
class RegressionModel(torch.nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for regression.
    It processes the (21, 8, 8) input and outputs a single float.
    """

    def __init__(self):
        super(RegressionModel, self).__init__()
        # Input: (batch_size, 21, 8, 8) -> 21 is the number of channels

        # Convolutional layers
        self.conv_layers = torch.nn.Sequential(
            # Conv1: Input 21 channels, Output 16 channels. Kernel size 3x3.
            torch.nn.Conv2d(in_channels=21, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )

        # Fully connected layers (for regression output)
        self.fc_layers = torch.nn.Sequential(
            # regression head
            torch.nn.Linear(32 * 8 * 8, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )  # Output a single float

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


def plot_losses(train_losses: list[float]) -> None:
    min_loss = min(train_losses)

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()

    # Annotate the minimum loss point
    min_epoch = train_losses.index(min_loss) + 1
    plt.scatter(min_epoch, min_loss, color="red")  # Mark the minimum point

    # add a text box with the min loss value, in top left corner of the plot
    plt.text(
        0.05,
        0.95,
        f"Min Loss: {min_loss:.4f} at Epoch {min_epoch}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    # Save the plot to output folder
    __basename__ = os.path.basename(__file__).replace(".py", "")
    plt_path = f"{__dirname__}/{__basename__}_training_loss.png"
    plt.savefig(plt_path)
    plt.close()


def normalize_evals_tensor(evals_tensor: torch.Tensor) -> torch.Tensor:
    return torch.tanh(evals_tensor / 10.0)


# Inverse transform function for later use
def denormalize_evals_tensor(normalized_evals_tensor: torch.Tensor) -> torch.Tensor:
    return torch.atanh(normalized_evals_tensor) * 10.0


def tensor_histogram_ascii(tensor: torch.Tensor, bins: int = 10, width: int = 50) -> str:
    hist, bin_edges = torch.histogram(tensor, bins=bins)
    hist = hist.cpu().numpy()
    bin_edges = bin_edges.cpu().numpy()

    max_count = hist.max()
    scale = width / max_count if max_count > 0 else 1

    histogram_lines = []
    for count, edge_start, edge_end in zip(hist, bin_edges[:-1], bin_edges[1:]):
        bar = "*" * int(count * scale)
        histogram_lines.append(f"{edge_start:6.2f} - {edge_end:6.2f} | {bar} ({count})")

    return "\n".join(histogram_lines)


def train() -> None:

    # ----------------------------------------------------------------

    # 1. Setup Data and Hyperparameters
    ################################################################
    # Check for GPU
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore

    # Hyperparameters
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    DATA_SIZE = 100_000

    # torch.manual_seed(42)

    boards_tensor, moves_tensor, evals_tensor = DatasetUtils.load_datasets(tensors_folder_path, max_file_count=4)
    evals_tensor = evals_tensor.view(-1, 1)  # Reshape to (N, 1)

    # Convert evals_tensor to numpy and display min/max
    evals_np = evals_tensor.cpu().numpy()
    print(f"evals_tensor: min={evals_np.min():.4f}, max={evals_np.max():.4f}")

    # Normalize evals to range [-1, 1] using sigmoid-like scaling
    evals_tensor = normalize_evals_tensor(evals_tensor)

    print("Evals Tensor Histogram:")
    print(tensor_histogram_ascii(evals_tensor, bins=20, width=50))

    # Convert evals_tensor to numpy and display min/max
    evals_np = evals_tensor.cpu().numpy()
    print(f"normalized evals_tensor: min={evals_np.min():.4f}, max={evals_np.max():.4f}")

    # downsample for quicker testing
    boards_tensor = boards_tensor[:DATA_SIZE]
    moves_tensor = moves_tensor[:DATA_SIZE]
    evals_tensor = evals_tensor[:DATA_SIZE]

    print(DatasetUtils.dataset_summary(boards_tensor, moves_tensor, evals_tensor))
    print(f"Evals tensor shape: {evals_tensor.shape}, dtype: {evals_tensor.dtype}")

    # Create Dataset and DataLoader
    dataset = torch.utils.data.TensorDataset(boards_tensor, evals_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ----------------------------------------------------------------

    # 2. Instantiate Model, Loss, and Optimizer
    ################################################################
    model = RegressionModel().to(device)
    # criterion = torch.nn.MSELoss()  # Mean Squared Error is common for regression
    criterion = torch.nn.L1Loss()  # Mean Absolute Error is common for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ----------------------------------------------------------------
    # Add a learning rate scheduler to reduce LR over time
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, threshold=0.05)
    early_stopper = EarlyStopper(patience=50, threshold=0.001)

    # 3. Training Loop
    ################################################################
    print(f"Starting training on {device}...")
    model.train()  # Set the model to training mode

    train_losses = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0

        for batch_index, (board_inputs, eval_outputs) in enumerate(tqdm.tqdm(dataloader, ncols=80)):
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
        training_loss = running_loss / len(dataloader) if len(dataloader) > 0 else float("nan")
        print(f"Epoch {epoch+1} lr={scheduler.get_last_lr()[0]} Loss: {training_loss:.8f}")

        # Plot training loss
        train_losses.append(training_loss)
        plot_losses(train_losses)

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
        test_input = boards_tensor[i : i + 1].to(device)  # Use an actual sample from the dataset

        # Disable gradient calculation for inference
        with torch.no_grad():
            prediction = model(test_input)

        unnormalized_prediction = denormalize_evals_tensor(prediction)
        unnormalized_actual = denormalize_evals_tensor(evals_tensor[i : i + 1])
        print(
            f"Example Prediction for a single input: {unnormalized_prediction.item():-3.4f} Actual eval: {unnormalized_actual.item():4.4f} delta={abs(unnormalized_prediction.item() - unnormalized_actual.item()):.4f}"
        )


###############################################################################
#   Main Entry Point
#
if __name__ == "__main__":
    train()
