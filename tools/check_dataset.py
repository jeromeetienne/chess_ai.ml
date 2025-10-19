#!/usr/bin/env python3

# stdlib imports
import os

# pip imports
import chess.polyglot
import argparse

# local imports
from src.libs.chess_extra import ChessExtra
from src.encoding.board_encoding import BoardEncoding
from src.utils.pgn_utils import PGNUtils
from src.utils.dataset_utils import DatasetUtils


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.abspath(os.path.join(__dirname__, "..", "data"))
output_folder_path = os.path.abspath(os.path.join(__dirname__, "..", "output"))
pgn_folder_path = os.path.abspath(os.path.join(data_folder_path, "pgn"))
tensor_folder_path = os.path.abspath(os.path.join(output_folder_path, "pgn_tensors"))

###############################################################################
#   Main entry point
#
if __name__ == "__main__":
    # Parse command line arguments
    argParser = argparse.ArgumentParser(
        description="Check the integrity of the dataset by comparing PGN files to their tensor representations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argParser.add_argument("--max_files_count", "-fc", type=int, default=10, help="Maximum number of PGN files to process. 0 for no limit.")
    argParser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")
    args = argParser.parse_args()

    # Load polyglot opening book
    polyglot_path = os.path.join(data_folder_path, "polyglot/lichess_pro_books/lpb-allbook.bin")
    polyglot_reader = chess.polyglot.open_reader(polyglot_path)

    ###############################################################################
    #   Check all PGN files against their tensor representations
    #
    pgn_paths = PGNUtils.get_pgn_paths()
    # truncate file_pgn_paths to max_files_count
    if args.max_files_count > 0:
        pgn_paths = pgn_paths[: args.max_files_count]

    for pgn_path in pgn_paths:
        difference_count = DatasetUtils.check_tensor_vs_pgn(pgn_path, polyglot_reader, verbose=args.verbose)
        if difference_count == 0:
            print(f"No differences found for {os.path.basename(pgn_path)}")
        else:
            print(f"Found {difference_count} differences for {os.path.basename(pgn_path)}")
