#!/usr/bin/env python3

# stdlib imports
import argparse

# local imports
from src.build_dataset import DatasetBuilderCommand

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
        "--max-files-count", type=int, default=10, help="Maximum number of PGN files to process."
    )
    parser.add_argument(
        "--max-games-count",
        type=int,
        default=0,
        help="Maximum number of games to process. Use 0 for no limit.",
    )
    args = parser.parse_args()

    DatasetBuilderCommand.build_dataset(max_files_count=args.max_files_count, max_games_count=args.max_games_count)
