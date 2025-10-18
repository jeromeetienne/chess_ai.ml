#!/usr/bin/env python3

# stdlib imports
import os

# pip imports
import argparse
import chess

# local imports
from src.libs.chess_extra import ChessExtra
from src.encoding.board_encoding import BoardEncoding
from src.encoding.move_encoding import MoveEncoding
from src.utils.dataset_utils import DatasetUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.join(__dirname__, "..", "output")
tensors_folder_path = os.path.join(output_folder_path, "pgn_tensors")

if __name__ == "__main__":
    # Parse command line arguments
    argParser = argparse.ArgumentParser(description="Build eval tensors from fishtest pgn files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argParser.add_argument("--max-files-count", "-fc", type=int, default=10, help="Maximum number of PGN files to process. 0 for no limit.")
    args = argParser.parse_args()

    # Load datasets
    boards_tensor, moves_tensor, evals_tensor, moves_index = DatasetUtils.load_datasets(tensors_folder_path, args.max_files_count)

    for i, (board_tensor, move_tensor, eval_tensor, move_index) in enumerate(zip(boards_tensor, moves_tensor, evals_tensor, moves_index)):
        board = BoardEncoding.board_from_tensor(board_tensor)
        move_uci = MoveEncoding.decode_move_tensor_classindex(move_tensor, board.turn)
        eval = eval_tensor.item()

        print(f"Board: {'white' if board.turn == chess.WHITE else 'black'} to move - {move_uci}")
        print(board)
        print(f"Moves idx: game {move_index['game_idx']} move {move_index['move_idx']} - Eval value: {eval}")
        print(f'Endgame {"yes" if ChessExtra.is_endgame(board) else "no"}')
        print("-" * 40)
