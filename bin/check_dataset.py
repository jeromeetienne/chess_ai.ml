#!/usr/bin/env python3

# stdlib imports
import os

# pip imports
import chess.polyglot
import argparse

# local imports
from src.libs.chess_extra import ChessExtra
from src.libs.encoding import Encoding
from src.utils.pgn_utils import PGNUtils
from src.utils.dataset_utils import DatasetUtils


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "..", "data")
pgn_folder_path = os.path.join(data_folder_path, "pgn")
tensor_folder_path = os.path.join(data_folder_path, "pgn_tensors")

###############################################################################
#   Main entry point
#
if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="Check the integrity of the dataset by comparing PGN files to their tensor representations.")
    args = argParser.parse_args()

    # Load polyglot opening book
    polyglot_path = os.path.join(data_folder_path, "polyglot/lichess_pro_books/lpb-allbook.bin")
    polyglot_reader = chess.polyglot.open_reader(polyglot_path)

    ###############################################################################
    #   Check all PGN files against their tensor representations
    #
    for pgn_path in PGNUtils.get_pgn_paths():
        difference_count = DatasetUtils.check_tensor_from_pgn(pgn_path, polyglot_reader, verbose=True)
        if difference_count == 0:
            print(f"No differences found for {os.path.basename(pgn_path)}")
        else:
            print(f"Found {difference_count} differences for {os.path.basename(pgn_path)}")


