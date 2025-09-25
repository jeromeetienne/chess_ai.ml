# stdlib imports
import os
import sys
import time

# pip imports
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# local imports
from .libs.chess_dataset import ChessDataset
from .libs.chess_model import ChessModel
from .libs.early_stopper import EarlyStopper
from .libs.utils import Utils
from .libs.io_utils import IOUtils

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = f"{__dirname__}/../output/"


class TrainCommand:
    ###############################################################################
    # Train a chess model using PyTorch.
    ###############################################################################
    @staticmethod
    def train(num_epochs: int = 20, batch_size: int = 2048, learning_rate: float = 0.001, train_test_split_ratio: float = 0.7):

        # set random seed for reproducibility
        # torch.manual_seed(42)

        ###############################################################################
        # Load Dataset
        #

        # sanity check: ensure dataset exists else exit
        if not IOUtils.has_dataset(output_folder_path):
            print("Dataset not found. Please create a new one.")
            sys.exit(1)

        dataset_creation_start_time = time.time()
        # Load the dataset
        boards_tensor, moves_tensor, uci_to_classindex = IOUtils.load_dataset(folder_path=output_folder_path)
        boards_dataset = ChessDataset(boards_tensor, moves_tensor)

        print(f"Total boards in dataset: {len(boards_tensor)}")
        dataset_creation_elapsed_time = time.time() - dataset_creation_start_time
        print(f"Dataset creation/loading time: {dataset_creation_elapsed_time:.2f} seconds")

        num_classes = len(uci_to_classindex)
        print(f"Number of classes: {num_classes}")
        print(f"boards_tensor shape: {boards_tensor.shape}")
        print(f"best_move_tensor shape: {moves_tensor.shape}")

        print(Utils.dataset_summary(boards_tensor, moves_tensor, uci_to_classindex))

        ###############################################################################
        # Prepare data loaders
        #

        dataset_ratio_train = train_test_split_ratio
        dataset_ratio_validation = (1 - train_test_split_ratio) / 2
        dataset_ratio_test = (1 - train_test_split_ratio) / 2

        # Create the datasets for training, validation and testing by splitting the original dataset
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
            boards_dataset, [dataset_ratio_train, dataset_ratio_validation, dataset_ratio_test]
        )

        # Create dataloaders for training, validation and testing
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
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

        print(Utils.model_summary(model))

        ###############################################################################
        # Training
        #

        def train_one_epoch(model: nn.Module, dataloader: DataLoader) -> float:
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

        def validation_one_epoch(model: nn.Module, dataloader: DataLoader) -> float:
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

        def evaluate_model_accuracy(model: nn.Module, dataloader: DataLoader) -> float:
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

        ###########################################################################

        early_stopper = EarlyStopper(patience=10)
        for epoch_index in range(num_epochs):
            epoch_start_time = time.time()
            # Training the model
            avg_loss = train_one_epoch(model=model, dataloader=train_dataloader)

            # Validate the model on the validation set
            validation_loss = validation_one_epoch(model=model, dataloader=validation_dataloader)

            # Check for early stopping
            must_stop, must_save = early_stopper.early_stop(validation_loss)

            epoch_elapsed_time = time.time() - epoch_start_time
            print(f"Epoch {epoch_index + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {validation_loss:.4f}, Time: {epoch_elapsed_time:.2f}-sec {'(Saved)' if must_save else '(worst)'}")

            # Save the model if validation loss improved
            if must_save:
                # Save the model
                IOUtils.save_model(model, folder_path=output_folder_path)
                print(f"Model saved to {output_folder_path}")

                # Write a README file with training details
                readme_md = f"""# Chess Model Training
- Trained at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
- Model trained on {len(train_dataset)} game positions
- Number of unique moves: {num_classes}
- Number of epochs: {epoch_index + 1}
- Final Loss: {validation_loss:.4f}

# Model Summary
{Utils.model_summary(model)}
                """
                README_path = f"{output_folder_path}/README.md"
                with open(README_path, "w") as readme_file:
                    readme_file.write(readme_md)

                # Now test the model on the test set
                eval_accuracy = evaluate_model_accuracy(model, test_dataloader)
                print(f"Accuracy on test set: {eval_accuracy:.2f}%")


            # Stop training if no improvement for 'patience' epochs
            if must_stop:
                print(f"Early stopping triggered at epoch {epoch_index + 1}")
                break

        ##########################################################################################

        print("Training complete.")

        # Now test the model on the test set
        eval_accuracy = evaluate_model_accuracy(model, test_dataloader)
        print(f"Accuracy on test set: {eval_accuracy:.2f}%")
