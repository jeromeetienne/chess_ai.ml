# stdlib imports
import os
import sys
import time
import argparse

# pip imports
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# local imports
from libs.chess_dataset import ChessDataset
from libs.chess_model import ChessModel
from libs.utils import Utils
from libs.io_utils import IOUtils

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))


def train(num_epochs: int = 20, batch_size: int = 2048, learning_rate: float = 0.001, train_test_split_ratio: float = 0.7):
    ###############################################################################
    # Load Dataset
    #

    output_folder_path = f"{__dirname__}/../output/"

    # sanity check: ensure dataset exists else exit
    if not IOUtils.has_dataset(output_folder_path):
        print("Dataset not found. Creating a new one.")
        sys.exit(1)

    dataset_creation_start_time = time.time()
    # Load the dataset
    boards_tensor, best_move_tensor, uci_to_classindex = IOUtils.load_dataset(folder_path=output_folder_path)

    print(f"Total boards in dataset: {len(boards_tensor)}")
    dataset_creation_elapsed_time = time.time() - dataset_creation_start_time
    print(f"Dataset creation/loading time: {dataset_creation_elapsed_time:.2f} seconds")

    num_classes = len(uci_to_classindex)
    print(f"Number of classes: {num_classes}")
    print(f'boards_tensor shape: {boards_tensor.shape}')
    print(f'best_move_tensor shape: {best_move_tensor.shape}')

    ###############################################################################
    # Prepare data loaders
    #

    # Create train_dataset and train_dataloader with the first 80% of the data
    train_inputs = boards_tensor[0 : int(train_test_split_ratio * len(boards_tensor))]
    train_labels = best_move_tensor[0 : int(train_test_split_ratio * len(best_move_tensor))]
    train_dataset = ChessDataset(train_inputs, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create test_dataset and test_dataloader with the remaining 20% of the data
    test_inputs = boards_tensor[int(train_test_split_ratio * len(boards_tensor)) :]
    test_labels = best_move_tensor[int(train_test_split_ratio * len(best_move_tensor)) :]
    test_dataset = ChessDataset(test_inputs, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Check for GPU
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
    print(f"Pytorch computes on {device} device")

    # Model Initialization
    model = ChessModel(num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ###############################################################################
    # Display model summary

    print(model)

    # TODO put that in a library function

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:_}")

    print("\nTrainable parameters per layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:40s}: {param.numel():_}")

    ###############################################################################
    # Training
    #

    def train_one_epoch() -> float:
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm.tqdm(train_dataloader, ncols=80, desc="Training", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()

            outputs = model(inputs)  # Raw logits

            # Compute loss
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_dataloader)
        return average_loss

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        avg_loss = train_one_epoch()
        epoch_elapsed_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_elapsed_time:.2f}-sec")

        # Save the model
        IOUtils.save_model(model, folder_path=output_folder_path)
        print(f"Model saved to {output_folder_path}")

        # Write a README file with training details
        readme_md = f"""# Chess Model Training
        - Model trained on {len(train_inputs)} game positions
        - Number of unique moves: {num_classes}
        - Number of epochs: {num_epochs}
        - Final Loss: {avg_loss:.4f}
        - trainable_params: {trainable_params:_}
        - model: {model}
        """
        README_path = f"{output_folder_path}/README.md"
        with open(README_path, "w") as readme_file:
            readme_file.write(readme_md)

    ##########################################################################################

    print("Training complete.")

    def evaluate_model_accuracy(model: nn.Module, dataloader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    # Now test the model on the test set
    eval_accuracy = evaluate_model_accuracy(model, test_dataloader)
    print(f"Accuracy on test set: {eval_accuracy:.2f}%")


###############################################################################
###############################################################################
# 	 Main Entry Point
###############################################################################
###############################################################################

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train a chess model using PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training")
    parser.add_argument(
        "--train_test_split_ratio", type=float, default=0.7, help="Train/test split ratio (between 0 and 1)"
    )
    # --learning_rate 0.001 or -lr 0.001
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    args = parser.parse_args()

    if args.debug:
        print(f"Arguments: {args}")
        print("Debug mode is ON")

    # Call the train function
    train(num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, train_test_split_ratio=args.train_test_split_ratio)
