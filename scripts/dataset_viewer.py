#!/usr/bin/env python3

import argparse
import os

from src.libs.encoding import Encoding
from src.utils.dataset_utils import DatasetUtils
from src.utils.pgn_utils import PGNUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.join(__dirname__, "..", "output")
tensors_folder_path = os.path.join(output_folder_path, "pgn_tensors")

if __name__ == "__main__":
    # Parse command line arguments
    argParser = argparse.ArgumentParser(description="Build eval tensors from fishtest pgn files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argParser.add_argument("--max-files-count", "-fc", type=int, default=10, help="Maximum number of PGN files to process. 0 for no limit.")
    args = argParser.parse_args()

    boards_tensor, moves_tensor, evals_tensor = DatasetUtils.load_datasets(tensors_folder_path, args.max_files_count)

    for i, (board_tensor, move_tensor, eval_tensor) in enumerate(zip(boards_tensor, moves_tensor, evals_tensor)):
        board = Encoding.board_from_tensor(board_tensor)
        move_uci = Encoding.move_from_tensor(move_tensor, board.turn)
        eval = eval_tensor.item()

        print("Board:")
        print(board)
        print("Move (UCI):", move_uci)
        print("Eval:", eval)
        print("-" * 40)

