#!/usr/bin/env python3

# stdlib imports
import argparse
import os
import time

# pip imports
import tqdm
import torch

# from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# local imports
from src.pytorch_extra.early_stopper import EarlyStopper
from src.utils.model_utils import ModelUtils
from src.utils.dataset_utils import DatasetUtils
from src.utils.model_utils import ModelUtils
from src.utils.uci2class_utils import Uci2ClassUtils
from src.pytorch_extra.model_summary import ModelSummary

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.join(__dirname__, "..", "output")
data_folder_path = os.path.join(__dirname__, "..", "data")
model_folder_path = os.path.join(output_folder_path, "model")
tensors_folder_path = os.path.join(output_folder_path, "pgn_tensors")


class TrainCommand:

    ###############################################################################
    # Train a chess model using PyTorch.
    ###############################################################################
    @staticmethod
    def train(
        model_name=ModelUtils.MODEL_NAME.CHESS_MODEL_CONV2D,
        max_epoch_count: int = 20,
        batch_size: int = 2048,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        early_stopping_threshold: float = 0.01,
        scheduler_patience: int = 3,
        scheduler_threshold: float = 0.1,
        train_test_split_ratio: float = 0.7,
        max_file_count: int = 15,
        random_seed: int | None = None,
        verbose: bool = True,
    ) -> float:
        """
        Train a chess model using PyTorch.

        Args:
            model_name (str): Name of the model architecture to use.
            max_epoch_count (int): Maximum number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            train_test_split_ratio (float): Ratio of training data to total data.
            max_file_count (int): Maximum number of PGN files to process. 0 for no limit.
            random_seed (int | None): Random seed for reproducibility. None for random seed.
            verbose (bool): Whether to print verbose output.
        Returns:
            model_accuracy_final (float): Final accuracy of the trained model.
        """

        # set random seed for reproducibility
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # =============================================================================
        # Print arguments
        # =============================================================================
        if verbose:
            print("Training parameters:")
            print(f"- Model name: {model_name}")
            print(f"- Max epoch count: {max_epoch_count}")
            print(f"- Batch size: {batch_size}")
            print(f"- Learning rate: {learning_rate}")
            print(f"- Train/test split ratio: {train_test_split_ratio}")
            print(f"- Max file count: {max_file_count if max_file_count > 0 else 'No limit'}")
            print(f"- Random seed: {random_seed}")

        # =============================================================================
        # Load the dataset
        # =============================================================================

        # Load the dataset
        boards_tensor, moves_tensor, evals_tensor, moves_index = DatasetUtils.load_datasets(tensors_folder_path, max_file_count, verbose=verbose)

        if verbose:
            print(DatasetUtils.dataset_summary(boards_tensor, moves_tensor, evals_tensor))

        # FIXME why is this needed on evals_tensor but not moves_tensor... because
        # - evals is a regression and use SmoothL1Loss as loss function ?
        # - moves is a classification problem and uses CrossEntropyLoss ?

        # Reshape evals_tensor to (N, 1) to match the expected input shape for the model
        evals_tensor = evals_tensor.reshape(-1, 1)  # Reshape to (N, 1)
        # moves_tensor = moves_tensor.reshape(-1, 1)  # Reshape to (N,)
        num_classes = Uci2ClassUtils.get_num_classes()

        # =============================================================================
        # Prepare data loaders
        # =============================================================================

        # Convert evals_tensor to numpy and display min/max
        # evals_np = evals_tensor.cpu().numpy()
        # print(f"evals_tensor: min={evals_np.min():.4f}, max={evals_np.max():.4f}")

        # Normalize evals to range [-1, 1] using sigmoid-like scaling
        # evals_tensor = DatasetUtils.normalize_evals_tensor(evals_tensor)

        # evals_means = evals_tensor.mean().item()
        # evals_stds = evals_tensor.std().item()
        # evals_tensor = (evals_tensor - evals_means) / (evals_stds + 1e-8)  # avoid division by zero
        # evals_tensor = torch.tanh(evals_tensor/3)

        # print("Evals Tensor Histogram:")
        # print(DatasetUtils.tensor_histogram_ascii(evals_tensor, bins=20, width=50))

        # Convert evals_tensor to numpy and display min/max
        evals_np = evals_tensor.cpu().numpy()
        if verbose:
            print(f"normalized evals_tensor: min={evals_np.min():.4f}, max={evals_np.max():.4f}")

        # =============================================================================
        # Prepare datasets and dataloaders
        # =============================================================================

        dataset_ratio_train = train_test_split_ratio
        dataset_ratio_validation = (1 - train_test_split_ratio) / 2
        dataset_ratio_test = (1 - train_test_split_ratio) / 2

        # Create the datasets for training, validation and testing by splitting the original dataset
        boards_dataset = torch.utils.data.TensorDataset(boards_tensor, moves_tensor, evals_tensor)
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
            boards_dataset, [dataset_ratio_train, dataset_ratio_validation, dataset_ratio_test]
        )

        # Create dataloaders for training, validation and testing
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Check for GPU
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
        # print(f"Pytorch computes on {device} device")

        # =============================================================================
        # Create the model
        # =============================================================================

        model = ModelUtils.create_model(model_name)
        model = model.to(device)

        # =============================================================================
        # Setup training components: loss functions, optimizer, scheduler, early stopper
        # =============================================================================

        # Losses: classification + regression
        criterion_cls = torch.nn.CrossEntropyLoss()
        # criterion_reg = torch.nn.MSELoss()
        # criterion_reg = torch.nn.L1Loss()
        criterion_reg = torch.nn.SmoothL1Loss()

        # use Adam optimizer to update model weights
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Add a learning rate scheduler to reduce LR over time
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=scheduler_patience, threshold=scheduler_threshold)
        # Initialize early stopper to stop training if no improvement for 'patience' epochs
        early_stopper = EarlyStopper(patience=early_stopping_patience, threshold=early_stopping_threshold)

        ###############################################################################
        # Display model summary

        if verbose:
            print(ModelSummary.to_string(model))

        # =============================================================================
        # Training loop
        # =============================================================================
        valid_losses: list[float] = []
        valid_cls_losses: list[float] = []
        valid_reg_losses: list[float] = []
        train_losses: list[float] = []
        train_cls_losses: list[float] = []
        train_reg_losses: list[float] = []

        # initial loss weights - To tune based on the losses observed during training
        loss_cls_weight = 0.05
        loss_reg_weight = 1.95

        for epoch_index in range(max_epoch_count):
            epoch_start_time = time.time()

            # =============================================================================
            # Dynamic loss weighting
            # =============================================================================
            # Dynamic loss weighting based on recent training losses
            # TODO put that in a function
            if True:
                loss_count = min(10_000, len(train_cls_losses))
                assert len(train_cls_losses) == len(train_reg_losses), "train_cls_losses and train_reg_losses must have the same length"
                loss_cls_weight = 1.0 / (sum(train_cls_losses[-loss_count:]) / loss_count + 1e-8) if loss_count > 1 else loss_cls_weight
                loss_reg_weight = 1.0 / (sum(train_reg_losses[-loss_count:]) / loss_count + 1e-8) if loss_count > 1 else loss_reg_weight
                # Normalize weights to keep total weight = 2.0
                total_weight = loss_cls_weight + loss_reg_weight
                loss_cls_weight = (loss_cls_weight / total_weight) * 2.0
                loss_reg_weight = (loss_reg_weight / total_weight) * 2.0
            else:
                loss_cls_weight = 0.5
                loss_reg_weight = 2 - loss_cls_weight

            # =============================================================================
            #
            # =============================================================================
            # Training the model
            train_loss, train_cls_loss, train_reg_loss = TrainCommand.train_one_epoch(
                model, train_dataloader, optimizer, criterion_cls, loss_cls_weight, criterion_reg, loss_reg_weight, device, verbose=verbose
            )

            # Validate the model on the validation set
            valid_loss, valid_cls_loss, valid_reg_loss = TrainCommand.validation_one_epoch(
                model, valid_dataloader, criterion_cls, loss_cls_weight, criterion_reg, loss_reg_weight, device
            )
            epoch_elapsed_time = time.time() - epoch_start_time

            # =============================================================================
            # End of epoch
            # =============================================================================

            # Step the scheduler
            scheduler.step(valid_loss)

            # update training and validation losses array
            train_losses.append(train_loss)
            train_cls_losses.append(train_cls_loss)
            train_reg_losses.append(train_reg_loss)
            valid_losses.append(valid_loss)
            valid_cls_losses.append(valid_cls_loss)
            valid_reg_losses.append(valid_reg_loss)

            # Check for early stopping
            must_stop, must_save = early_stopper.step(valid_loss)

            # Print epoch summary
            if verbose:
                print(f"Epoch {epoch_index + 1}/{max_epoch_count}", end=" | ")
                print(f"lr={scheduler.get_last_lr()[0]} badEpoch={scheduler.num_bad_epochs}/{scheduler.patience}", end=" | ")
                print(f"early-stop badEpoch={early_stopper.bad_epoch_count}/{early_stopper.patience}", end=" | ")
                print(f"Train Loss: {train_loss:.4f} (cls={(train_cls_loss*loss_cls_weight):.4f} reg={(train_reg_loss*loss_reg_weight):.4f})", end=" | ")
                print(f"Valid Loss: {valid_loss:.4f} (cls={(valid_cls_loss*loss_cls_weight):.4f} reg={(valid_reg_loss*loss_reg_weight):.4f})", end=" | ")
                print(f"Time: {epoch_elapsed_time:.2f}-sec {'(Saved)' if must_save else '(worst)'}")

            # Plot training and validation loss
            TrainCommand.plot_losses(
                train_losses, train_cls_losses, train_reg_losses, loss_cls_weight, valid_losses, valid_cls_losses, valid_reg_losses, loss_reg_weight
            )

            # honor must_save: Save the model if validation loss improved
            if must_save:
                # Save the model
                ModelUtils.save_model(model, folder_path=model_folder_path)

                # Save training report
                TrainCommand.save_training_report(train_dataset, validation_dataset, test_dataset, num_classes, epoch_index, valid_loss, model)

                # Now test the model on the test set
                eval_cls_accuracy, eval_reg_mae = TrainCommand.evaluate_model(model, test_dataloader, device)
                if verbose:
                    print(f"Test dataset: classification accuracy: {eval_cls_accuracy:.2f}% - regression MAE: {eval_reg_mae:.4f}")

            # honor must_stop: Stop training if no improvement for 'patience' epochs
            if must_stop:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch_index + 1}")
                break

        ##########################################################################################

        if verbose:
            print("Training complete.")

        # Now test the model on the test set
        eval_cls_accuracy, eval_reg_mae = TrainCommand.evaluate_model(model, test_dataloader, device)
        model_accuracy_final = loss_cls_weight * eval_cls_accuracy + loss_reg_weight * eval_reg_mae
        if verbose:
            print(
                f"Test dataset: classification accuracy: {eval_cls_accuracy:.2f}% - regression MAE: {eval_reg_mae:.4f} - weighted sum: {model_accuracy_final:.4f}"
            )
        return model_accuracy_final

    # =============================================================================
    # Helper methods
    # =============================================================================

    @staticmethod
    def plot_losses(
        train_losses: list[float],
        train_cls_losses: list[float],
        train_reg_losses: list[float],
        loss_cls_weight: float,
        valid_losses: list[float],
        valid_cls_losses: list[float],
        valid_reg_losses: list[float],
        loss_reg_weight: float,
    ):
        epochs = range(1, len(train_losses) + 1)
        figure, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
        axes_tot_loss = axes[0]
        axes_cls_loss = axes[1]
        axes_reg_loss = axes[2]

        # =============================================================================
        # Total Loss
        # =============================================================================
        axes_tot_loss.plot(epochs, train_losses, label="Training Loss")
        axes_tot_loss.plot(epochs, valid_losses, label="Validation Loss")
        axes_tot_loss.set_ylabel("Total Loss")
        axes_tot_loss.set_title(f"Total Loss per Epoch (cls x {loss_cls_weight:.2f} + reg x {loss_reg_weight:.2f})")
        axes_tot_loss.legend()

        # Annotate the minimum loss point
        tot_loss_min = min(valid_losses)
        tot_loss_min_epoch = valid_losses.index(tot_loss_min) + 1
        axes_tot_loss.scatter(tot_loss_min_epoch, tot_loss_min, color="red")  # Mark the minimum point

        # add a text box with the min loss value, in top left corner of the plot
        axes_tot_loss.text(
            0.05,
            0.95,
            f"Min Loss: {tot_loss_min:.4f} at Epoch {tot_loss_min_epoch}",
            transform=axes_tot_loss.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        # =============================================================================
        # Classification Loss
        # =============================================================================
        axes_cls_loss.plot(epochs, train_cls_losses, label=f"Training Class Loss")
        axes_cls_loss.plot(epochs, valid_cls_losses, label="Validation Class Loss")
        axes_cls_loss.set_ylabel("Class Loss")
        axes_cls_loss.set_title(f"Classification Loss per Epoch")
        axes_cls_loss.legend()

        # Annotate the minimum loss point
        cls_loss_min = min(valid_cls_losses)
        cls_loss_min_epoch = valid_cls_losses.index(cls_loss_min) + 1
        axes_cls_loss.scatter(cls_loss_min_epoch, cls_loss_min, color="red")  # Mark the minimum point

        # add a text box with the min loss value, in top left corner of the plot
        axes_cls_loss.text(
            0.05,
            0.95,
            f"Min Loss: {cls_loss_min:.4f} at Epoch {cls_loss_min_epoch}",
            transform=axes_cls_loss.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        # =============================================================================
        # Regression Loss
        # =============================================================================
        axes_reg_loss.plot(epochs, train_reg_losses, label=f"Training Regression Loss")
        axes_reg_loss.plot(epochs, valid_reg_losses, label="Validation Regression Loss")
        axes_reg_loss.set_xlabel("Epoch")
        axes_reg_loss.set_ylabel("Regression Loss")
        axes_reg_loss.set_title(f"Regression Loss per Epoch")
        axes_reg_loss.legend()

        # Annotate the minimum loss point
        reg_loss_min = min(valid_reg_losses)
        reg_loss_min_epoch = valid_reg_losses.index(reg_loss_min) + 1
        axes_reg_loss.scatter(reg_loss_min_epoch, reg_loss_min, color="red")  # Mark the minimum point

        # add a text box with the min loss value, in top left corner of the plot
        axes_reg_loss.text(
            0.05,
            0.95,
            f"Min Loss: {reg_loss_min:.4f} at Epoch {reg_loss_min_epoch}",
            transform=axes_reg_loss.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        # plt.tight_layout()
        # Save the plot to output folder
        plt_path = f"{model_folder_path}/training_validation_loss.png"
        plt.savefig(plt_path)
        plt.close()

    @staticmethod
    def save_training_report(
        train_dataset: torch.utils.data.Subset,
        validation_dataset: torch.utils.data.Subset,
        test_dataset: torch.utils.data.Subset,
        num_classes: int,
        epoch_index: int,
        validation_loss: float,
        model: torch.nn.Module,
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
{ModelSummary.to_string(model)}
```
        """
        report_path = f"{model_folder_path}/TRAINING_REPORT.md"
        with open(report_path, "w") as report_file:
            report_file.write(file_content)
        # print(f"Training report saved to {README_path}")

    @staticmethod
    def train_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion_cls: torch.nn.Module,
        loss_cls_weight: float,
        criterion_reg: torch.nn.Module,
        loss_reg_weight: float,
        device: str,
        verbose: bool = True,
    ) -> tuple[float, float, float]:
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_reg_loss = 0.0

        for boards_inputs, moves_outputs, evals_outputs in tqdm.tqdm(dataloader, ncols=80, desc="Training", unit="batch", disable=not verbose):
            # Move tensors to the appropriate device
            boards_inputs = boards_inputs.to(device)
            moves_outputs = moves_outputs.to(device)
            evals_outputs = evals_outputs.to(device)

            # reset gradients
            optimizer.zero_grad()

            # Model forward pass
            moves_preds, evals_preds = model(boards_inputs)

            # classification loss expects shape (N, C) and targets (N,)
            cls_loss = criterion_cls(moves_preds, moves_outputs)

            # regression loss: ensure shapes match (N,1)
            reg_loss = criterion_reg(evals_preds, evals_outputs)

            # total loss = weighted sum
            loss = loss_cls_weight * cls_loss + loss_reg_weight * reg_loss

            # Backpropagation
            loss.backward()

            if False:
                # compute total grad norm (before clipping)
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm**0.5
                print(f"grad_norm_before_clip = {total_norm:.4f}")
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2.0)

            optimizer.step()

            running_loss += loss.item()
            running_cls_loss += cls_loss.item()
            running_reg_loss += reg_loss.item()

        train_loss = running_loss / len(dataloader)
        train_cls_loss = running_cls_loss / len(dataloader)
        train_reg_loss = running_reg_loss / len(dataloader)

        return train_loss, train_cls_loss, train_reg_loss

    @staticmethod
    def validation_one_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion_cls: torch.nn.Module,
        loss_cls_weight: float,
        criterion_reg: torch.nn.Module,
        loss_reg_weight: float,
        device: str,
    ) -> tuple[float, float, float]:
        model.eval()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_reg_loss = 0.0

        with torch.no_grad():
            for boards_inputs, moves_outputs, evals_outputs in dataloader:
                # Move tensors to the appropriate device
                boards_inputs = boards_inputs.to(device)
                moves_outputs = moves_outputs.to(device)
                evals_outputs = evals_outputs.to(device)

                # Model forward pass
                moves_preds, evals_preds = model(boards_inputs)

                # classification loss expects shape (N, C) and targets (N,)
                cls_loss = criterion_cls(moves_preds, moves_outputs)

                # regression loss: ensure shapes match (N,1)
                reg_loss = criterion_reg(evals_preds, evals_outputs)

                # total loss = weighted sum
                loss = loss_cls_weight * cls_loss + loss_reg_weight * reg_loss

                # update running losses
                running_loss += loss.item()
                running_cls_loss += cls_loss.item()
                running_reg_loss += reg_loss.item()

        # compute average losses
        valid_loss = running_loss / len(dataloader)
        valid_cls_loss = running_cls_loss / len(dataloader)
        valid_reg_loss = running_reg_loss / len(dataloader)

        return valid_loss, valid_cls_loss, valid_reg_loss

    @staticmethod
    def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str) -> tuple[float, float]:
        """
        Evaluate the classification accuracy of the model on the given dataloader.
        """
        # FIXME this function is buggy ?
        model.eval()
        accuracy_correct = 0
        accuracy_total = 0
        total_mae = 0.0
        total_samples = 0
        with torch.no_grad():
            for boards_tensor, moves_tensor, evals_tensor in dataloader:
                # Move tensors to the appropriate device
                boards_tensor = boards_tensor.to(device)
                moves_tensor = moves_tensor.to(device)
                evals_tensor = evals_tensor.to(device)

                # Model forward pass
                move_logits, eval_pred = model(boards_tensor)

                _, move_predictions = torch.max(move_logits, 1)  # Get the index of the max log-probability
                accuracy_total += moves_tensor.size(0)
                accuracy_correct += (move_predictions == moves_tensor).sum().item()

                # Compute Mean Absolute Error (MAE)
                mae = torch.abs(eval_pred - evals_tensor).sum().item()
                total_mae += mae
                total_samples += evals_tensor.size(0)

        mean_accuracy = 100 * accuracy_correct / accuracy_total
        average_mae = total_mae / total_samples

        return mean_accuracy, average_mae


###############################################################################
#   Main Entry Point
#
if __name__ == "__main__":
    # Parse command line arguments
    argParser = argparse.ArgumentParser(
        description="Train a chess model using PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argParser.add_argument("--max_epoch", "-me", type=int, default=100, help="Number of training epochs")
    argParser.add_argument("--batch_size", "-bs", type=int, default=2048, help="Batch size for training")
    argParser.add_argument("--train_test_split_ratio", "-ts", type=float, default=0.7, help="Train/test split ratio (between 0 and 1)")
    argParser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate for the optimizer")
    argParser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    argParser.add_argument("--max-files-count", "-fc", type=int, default=10, help="Maximum number of PGN files to process. 0 for no limit.")
    argParser.add_argument(
        "--model_name",
        "-mn",
        type=str,
        default=ModelUtils.MODEL_NAME.CHESS_MODEL_CONV2D,
        choices=ModelUtils.get_supported_models(),
        help="Model architecture to use for training",
    )
    args = argParser.parse_args()

    if args.debug:
        print(f"Arguments: {args}")
        print("Debug mode is ON")

    # Call the train function
    TrainCommand.train(
        model_name=args.model_name,
        max_epoch_count=args.max_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_test_split_ratio=args.train_test_split_ratio,
        max_file_count=args.max_files_count,
    )
