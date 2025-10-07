#!/usr/bin/env python3

# stdlib imports
import argparse

# local imports
from src.train import TrainCommand

###############################################################################
#   Main Entry Point
#
if __name__ == "__main__":
    # Parse command line arguments
    argParser = argparse.ArgumentParser(
        description="Train a chess model using PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argParser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    argParser.add_argument("--batch_size", "-bs", type=int, default=2048, help="Batch size for training")
    argParser.add_argument("--train_test_split_ratio", "-ts", type=float, default=0.7, help="Train/test split ratio (between 0 and 1)")
    argParser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate for the optimizer")
    argParser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    argParser.add_argument("--max-files-count", "-fc", type=int, default=10, help="Maximum number of PGN files to process. 0 for no limit.")
    argParser.add_argument(
        "--model_name",
        "-mn",
        type=str,
        default="ChessModelConv2d",
        choices=["ChessModelConv2d", "ChessModelResNet", "AlphaZeroNet"],
        help="Model architecture to use for training",
    )
    args = argParser.parse_args()

    if args.debug:
        print(f"Arguments: {args}")
        print("Debug mode is ON")

    # Call the train function
    TrainCommand.train(
        model_name=args.model_name,
        max_epoch_count=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_test_split_ratio=args.train_test_split_ratio,
        max_file_count=args.max_files_count,
    )
