# stdlib imports
import argparse

# local imports
from src.commands.train import TrainCommand

###############################################################################
#   Main Entry Point
#
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train a chess model using PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("--train_test_split_ratio", type=float, default=0.7, help="Train/test split ratio (between 0 and 1)")
    # --learning_rate 0.001 or -lr 0.001
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    args = parser.parse_args()

    if args.debug:
        print(f"Arguments: {args}")
        print("Debug mode is ON")

    # Call the train function
    TrainCommand.train(num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, train_test_split_ratio=args.train_test_split_ratio)
