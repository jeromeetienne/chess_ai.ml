# stdlib imports
import os
import sys
import time
import typing

# pip imports
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# local imports
from .libs.chess_dataset import ChessDataset
from .libs.chess_model import ChessModel
from .libs.early_stopper import EarlyStopper
from .libs.utils import Utils
from .libs.io_utils import IOUtils

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = f"{__dirname__}/../output"


class TrainCommand:

    @staticmethod
    def plot_losses(train_losses: list, validation_losses: list):
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train vs Validation Loss per Epoch")
        plt.legend()
        # Save the plot to output folder
        plt_path = f"{output_folder_path}/training_validation_loss.png"
        plt.savefig(plt_path)
        plt.close()
        # print(f"Training and validation loss plot saved to {plt_path}")

    @staticmethod
    def save_training_report(
        train_dataset: torch.utils.data.Subset,
        validation_dataset: torch.utils.data.Subset,
        test_dataset: torch.utils.data.Subset,
        num_classes: int,
        epoch_index: int,
        validation_loss: float,
        model: nn.Module,
    ):
        file_content = f"""# Chess Model Training
- Trained at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
- Model trained on {len(train_dataset)} game positions
- Model validated on {len(validation_dataset)} game positions
- Model tested on {len(test_dataset)} game positions
- Number of unique moves: {num_classes}
- Number of epochs: {epoch_index + 1}
- Final validation loss: {validation_loss:.4f}

# Model Summary
```
{Utils.model_summary(model)}
```
        """
        report_path = f"{output_folder_path}/TRAINING_REPORT.md"
        with open(report_path, "w") as report_file:
            report_file.write(file_content)
        # print(f"Training report saved to {README_path}")

    @staticmethod
    def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module, device: str) -> float:
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm.tqdm(dataloader, ncols=80, desc="Training", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()

            outputs = model(inputs)  # Raw logits

            # Compute loss
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(dataloader)
        return average_loss

    @staticmethod
    def validation_one_epoch(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: str) -> float:
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = loss_fn(outputs, labels)
                running_loss += loss.item()

        validation_loss = running_loss / len(dataloader)
        return validation_loss

    @staticmethod
    def evaluate_model_accuracy(model: nn.Module, dataloader: DataLoader, device: str) -> float:
        # FIXME this function is buggy ?
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    ###############################################################################
    # Train a chess model using PyTorch.
    ###############################################################################
    @staticmethod
    def train(num_epochs: int = 20, batch_size: int = 2048, learning_rate: float = 0.001, train_test_split_ratio: float = 0.7):

        # set random seed for reproducibility
        torch.manual_seed(42)

        ###############################################################################
        # Load Dataset
        #

        # sanity check: ensure dataset exists else exit
        if not IOUtils.has_dataset(output_folder_path):
            print("Dataset not found. Please create a new one.")
            sys.exit(1)

        # Load the dataset
        boards_tensor, moves_tensor, uci_to_classindex = IOUtils.load_dataset(folder_path=output_folder_path)
        print(Utils.dataset_summary(boards_tensor, moves_tensor, uci_to_classindex))

        num_classes = len(uci_to_classindex)

        ###############################################################################
        # Prepare data loaders
        #

        dataset_ratio_train = train_test_split_ratio
        dataset_ratio_validation = (1 - train_test_split_ratio) / 2
        dataset_ratio_test = (1 - train_test_split_ratio) / 2

        # Create the datasets for training, validation and testing by splitting the original dataset
        boards_dataset = ChessDataset(boards_tensor, moves_tensor)
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
            boards_dataset, [dataset_ratio_train, dataset_ratio_validation, dataset_ratio_test]
        )

        # Create dataloaders for training, validation and testing
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Check for GPU
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
        # print(f"Pytorch computes on {device} device")

        # Model Initialization
        input_shape: tuple[int, int, int] = typing.cast(tuple[int,int,int],boards_tensor.shape[1:])  # (channels, height, width)
        output_shape = (num_classes, )
        model = ChessModel(input_shape=input_shape, output_shape=output_shape).to(device)

        # use cross entropy loss for multi-class classification
        loss_fn = nn.CrossEntropyLoss()
        # use Adam optimizer to update model weights
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Add a learning rate scheduler to reduce LR over time
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=0.1)
        # Initialize early stopper to stop training if no improvement for 'patience' epochs
        early_stopper = EarlyStopper(patience=10, threshold=0.01)

        ###############################################################################
        # Display model summary

        print(Utils.model_summary(model))

        ###############################################################################
        # Training
        #
        validation_losses = []
        train_losses = []

        for epoch_index in range(num_epochs):
            epoch_start_time = time.time()
            # Training the model
            avg_loss = TrainCommand.train_one_epoch(model=model, dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn, device=device)

            # Validate the model on the validation set
            validation_loss = TrainCommand.validation_one_epoch(model=model, dataloader=validation_dataloader, loss_fn=loss_fn, device=device)
            epoch_elapsed_time = time.time() - epoch_start_time

            # Step the scheduler
            scheduler.step(validation_loss)

            # update training and validation losses array
            train_losses.append(avg_loss)
            validation_losses.append(validation_loss)

            # Check for early stopping
            must_stop, must_save = early_stopper.early_stop(validation_loss)

            # Print epoch summary            
            print(
                f"Epoch {epoch_index + 1}/{num_epochs}, lr={scheduler.get_last_lr()[0]} Training Loss: {avg_loss:.4f}, Validation Loss: {validation_loss:.4f}, Time: {epoch_elapsed_time:.2f}-sec {'(Saved)' if must_save else '(worst)'}"
            )

            # honor must_save: Save the model if validation loss improved
            if must_save:
                # Save the model
                IOUtils.save_model(model, folder_path=output_folder_path)
                # print(f"Model saved to {output_folder_path}")

                TrainCommand.save_training_report(train_dataset, validation_dataset, test_dataset, num_classes, epoch_index, validation_loss, model)

                # Plot training and validation loss
                TrainCommand.plot_losses(train_losses, validation_losses)

                # Now test the model on the test set
                eval_accuracy = TrainCommand.evaluate_model_accuracy(model, test_dataloader, device)
                print(f"Accuracy on test set: {eval_accuracy:.2f}%")

            # honor must_stop: Stop training if no improvement for 'patience' epochs
            if must_stop:
                print(f"Early stopping triggered at epoch {epoch_index + 1}")
                break

        ##########################################################################################

        print("Training complete.")

        # Now test the model on the test set
        eval_accuracy = TrainCommand.evaluate_model_accuracy(model, test_dataloader, device)
        print(f"Accuracy on test set: {eval_accuracy:.2f}%")
