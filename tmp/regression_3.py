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
from src.libs.encoding import Encoding
from src.utils.uci2class_utils import Uci2ClassUtils


__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.abspath(os.path.join(__dirname__, "..", "output"))
tensors_folder_path = os.path.join(output_folder_path, "pgn_tensors")


# Define the Model
################################################################
class DualHeadModel(torch.nn.Module):
    """
    Convolutional feature extractor with two heads:
    - classification head: outputs logits over move classes
    - regression head: outputs a single float evaluation
    """

    def __init__(self, num_classes: int):
        super(DualHeadModel, self).__init__()

        # feature extractor
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=Encoding.get_input_shape()[0], out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        flat_features = 128 * 8 * 8

        # classification head (move prediction)
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(flat_features, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes),
        )

        # regression head (eval prediction)
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(flat_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(torch.float32)
        features = self.conv_layers(x)
        move_logits = self.cls_head(features)
        eval_pred = self.reg_head(features)
        return move_logits, eval_pred


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
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 200
    DATA_SIZE = 100_000

    # torch.manual_seed(42)

    boards_tensor, moves_tensor, evals_tensor = DatasetUtils.load_datasets(tensors_folder_path, max_file_count=4)
    # moves_tensor is a scalar class index per sample (int)
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

    # Create Dataset and DataLoader (boards, moves, evals)
    # Ensure correct dtypes
    moves_tensor = moves_tensor.to(torch.long)
    dataset = torch.utils.data.TensorDataset(boards_tensor, moves_tensor, evals_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ----------------------------------------------------------------

    # 2. Instantiate Model, Loss, and Optimizer
    ################################################################
    # Build dual-head model
    num_classes = Encoding.get_output_shape()[0]
    model = DualHeadModel(num_classes=num_classes).to(device)

    # Losses: classification + regression
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_reg = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ----------------------------------------------------------------
    # Add a learning rate scheduler to reduce LR over time
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, threshold=0.05)
    early_stopper = EarlyStopper(patience=50, threshold=0.001)

    # 3. Training Loop
    ################################################################
    print(f"Starting training on {device}...")
    model.train()  # Set the model to training mode

    train_losses = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        running_cls_loss = 0.0
        running_reg_loss = 0.0

        for batch_index, (board_inputs, moves_targets, eval_outputs) in enumerate(tqdm.tqdm(dataloader, ncols=80)):
            # Move tensors to device
            board_inputs = board_inputs.to(device)
            moves_targets = moves_targets.to(device)
            eval_outputs = eval_outputs.to(device)

            optimizer.zero_grad()

            # Forward pass -> logits and regression prediction
            move_logits, eval_pred = model(board_inputs)

            # classification loss expects shape (N, C) and targets (N,)
            cls_loss = criterion_cls(move_logits, moves_targets)

            # regression loss: ensure shapes match (N,1)
            reg_loss = criterion_reg(eval_pred, eval_outputs)


            # total loss = weighted sum
            LOSS_REG_WEIGHT = 1.0
            LOSS_CLS_WEIGHT = 1.0
            loss = LOSS_CLS_WEIGHT * cls_loss + LOSS_REG_WEIGHT * reg_loss

            # Backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_cls_loss += cls_loss.item()
            running_reg_loss += reg_loss.item()

        # Print statistics every epoch
        training_loss = running_loss / len(dataloader) if len(dataloader) > 0 else float("nan")
        avg_cls = running_cls_loss / len(dataloader) if len(dataloader) > 0 else float("nan")
        avg_reg = running_reg_loss / len(dataloader) if len(dataloader) > 0 else float("nan")
        print(f"Epoch {epoch+1} lr={scheduler.get_last_lr()[0]} Loss: {training_loss:.8f} (cls={avg_cls:.6f} reg={avg_reg:.6f})")

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

    # show a few example predictions: move top-3 and eval
    for i in range(40):
        test_input = boards_tensor[i : i + 1].to(device)  # Use an actual sample from the dataset
        true_move = int(moves_tensor[i].item())
        true_eval = denormalize_evals_tensor(evals_tensor[i : i + 1]).item()

        # Disable gradient calculation for inference
        with torch.no_grad():
            move_logits, eval_pred = model(test_input)

        # classification top-k
        probs = torch.nn.functional.softmax(move_logits, dim=1)
        topk = torch.topk(probs, k=3, dim=1)
        topk_indices = topk.indices[0].cpu().numpy().tolist()
        topk_probs = topk.values[0].cpu().numpy().tolist()

        # map indices to uci
        color = Uci2ClassUtils.get_uci2class.__defaults__ if False else None
        # use moves_tensor's associated color via Encoding? We cannot infer color here; skip color-specific mapping and only show class indices

        unnormalized_pred_eval = denormalize_evals_tensor(eval_pred).item()

        print(f"Sample {i+1}: true_move_idx={true_move} true_eval={true_eval:4.4f} pred_eval={unnormalized_pred_eval:4.4f} top3_move_indices={topk_indices} top3_probs={[round(p,3) for p in topk_probs]}")


###############################################################################
#   Main Entry Point
#
if __name__ == "__main__":
    train()
