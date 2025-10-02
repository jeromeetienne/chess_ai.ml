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
    argParser = argparse.ArgumentParser(
        description="Build chess dataset from PGN files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argParser.add_argument(
        "--max-files-count", "-fc", type=int, default=15, help="Maximum number of PGN files to process. 0 for no limit."
    )
    args = argParser.parse_args()

    DatasetBuilderCommand.build_dataset(max_files_count=args.max_files_count)
