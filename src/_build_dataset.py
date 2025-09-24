# stdlib imports
import os
import time
import argparse

# local imports
from libs.utils import Utils
from libs.io_utils import IOUtils

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))

###############################################################################
# Create Dataset
###############################################################################
def build_dataset(max_files_count: int = 15, max_games_count: int = 7000):
    output_folder_path = f"{__dirname__}/../output/"
    dataset_creation_start_time = time.time()

    # Create dataset
    boards_tensor, best_move_tensor, uci_to_classindex = Utils.create_dataset(
        max_files_count=max_files_count, max_games_count=max_games_count
    )

    # Save the dataset for later
    IOUtils.save_dataset(boards_tensor, best_move_tensor, uci_to_classindex, folder_path=output_folder_path)

    # display elapsed time
    dataset_creation_elapsed_time = time.time() - dataset_creation_start_time
    print(f"Dataset creation/loading time: {dataset_creation_elapsed_time:.2f} seconds")

    # Dataset creation stats
    print(f"Total boards in dataset: {len(boards_tensor)}")
    print(f"Number of classes: {len(uci_to_classindex)}")


###############################################################################
###############################################################################
# 	 Main entry point
###############################################################################
###############################################################################

if __name__ == "__main__":
    # Parse command line arguments for max_files_count and max_games_count
    parser = argparse.ArgumentParser(
        description="Build chess dataset from PGN files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--max-files-count", type=int, default=15, help="Maximum number of PGN files to process."
    )
    parser.add_argument(
        "--max-games-count",
        type=int,
        default=7000,
        help="Maximum number of games to process. Use 0 for no limit.",
    )
    args = parser.parse_args()

    build_dataset(max_files_count=args.max_files_count, max_games_count=args.max_games_count)
