#!/usr/bin/env python3

# stdlib imports
import os
import time
import typing

# pip imports
import argparse
import chess
import chess.polyglot
import torch
import chess.engine

# local imports
from src.libs.chess_extra import ChessExtra
from src.encoding.board_encoding import BoardEncoding
from src.utils.dataset_utils import DatasetUtils
from src.utils.pgn_utils import PGNUtils
from bin.build_evals_stockfish import unify_engine_score


###############################################################################
#   Constants
#
__dirname__ = os.path.dirname(__file__)
data_folder_path = os.path.join(__dirname__, "../data")
output_folder_path = os.path.join(__dirname__, "../output")
pgn_folder_path = os.path.join(output_folder_path, "pgn_splits")
tensors_folder_path = os.path.join(output_folder_path, "pgn_tensors")

###############################################################################
#   Functions
#


def build_eval_array_from_pgn(
    pgn_path: str, polyglot_reader: chess.polyglot.MemoryMappedReader, skip_opening_book: bool = True, skip_endgame: bool = True
) -> list[float]:
    # play all pgn games of this file
    # - skip all positions already in the opening book
    # - extract the evaluation_score for each position and append it an array
    # - save the evals array as a tensor to dst_path

    pgn_evals: list[float] = []
    # parse the pgn file
    pgn_games = PGNUtils.pgn_file_to_games(pgn_path)

    # split the games into boards and moves
    pgn_boards, pgn_moves, pgn_moves_index = DatasetUtils.games_to_boards_moves(pgn_games, polyglot_reader)

    # iterate over all games
    for game in pgn_games:

        result_str = game.headers.get("Result", None)
        if result_str == "1-0":
            game_result_for_white = 1.0
        elif result_str == "0-1":
            game_result_for_white = -1.0
        elif result_str == "1/2-1/2":
            game_result_for_white = 0.0
        else:
            assert False, f"Unexpected game result: {result_str}"

        # iterate over all moves in the mainline, skipping positions in the opening book
        board = game.board()
        for node in game.mainline():
            board_before = board.copy(stack=False)

            # push the move to the board
            board.push(node.move)
            # skip if the position is in the opening book
            if skip_opening_book and polyglot_reader and ChessExtra.in_opening_book(board, polyglot_reader):
                continue

            # skip if is in endgame
            if skip_endgame and ChessExtra.is_endgame(board):
                continue

            # get the result probability for this move from the point of view of the player to move
            pgn_eval = game_result_for_white if board_before.turn == chess.WHITE else -game_result_for_white

            # append the evaluation score to the pgn_evals array
            pgn_evals.append(pgn_eval)

    # return the pgn_evals array
    return pgn_evals


def process_pgn_file(pgn_path: str, tensors_folder_path: str, polyglot_reader: chess.polyglot.MemoryMappedReader) -> None:
    # process a single pgn file
    basename = os.path.basename(pgn_path).replace(".pgn", "")

    # build dst_path
    dst_path = os.path.abspath(os.path.join(tensors_folder_path, f"{basename}{DatasetUtils.FILE_SUFFIX.EVALS}"))
    if os.path.exists(dst_path):
        print(f"{basename}.pgn already got a eval tensor, skipping.")
        return

    # log the event
    print(f"Processing {basename}.pgn...", end=" ", flush=True)

    time_start = time.perf_counter()
    # process the pgn file
    pgn_evals = build_eval_array_from_pgn(pgn_path, polyglot_reader)

    # convert pgn_evals to a torch tensor
    evals_tensor = torch.tensor(pgn_evals, dtype=BoardEncoding.EVAL_DTYPE)

    time_elapsed = time.perf_counter() - time_start

    # save evals_tensor to dst_path
    torch.save(evals_tensor, dst_path)

    # log the event
    print(f"Done in {time_elapsed:.2f} seconds. {len(evals_tensor) / time_elapsed:.2f} boards/s")


# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    argParser = argparse.ArgumentParser(
        description="""Build evaluation tensors for chess positions fishtest games using Stockfish engine.
The evals are from the point of view of the player to move, and are in the range [-1, 1], where 1 is a win for white, -1 is a win for black, and 0 is a draw.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argParser.add_argument("--max_files_count", "-fc", type=int, default=10, help="Maximum number of PGN files to process. 0 for no limit.")
    args = argParser.parse_args()
    # args = argParser.parse_args(['-fc', '4'])  # for testing only, remove this line for production

    # get all pgn paths
    pgn_paths = PGNUtils.get_pgn_paths()
    # keep only the first max-files-count pgn files
    if args.max_files_count > 0:
        pgn_paths = pgn_paths[: args.max_files_count]

    # Read the polyglot opening book
    polyglot_path = os.path.join(data_folder_path, "./polyglot/lichess_pro_books/lpb-allbook.bin")
    polyglot_reader = chess.polyglot.open_reader(polyglot_path)

    print(f"Processing {len(pgn_paths)} PGN files from {os.path.abspath(pgn_folder_path)}...")
    for pgn_path in pgn_paths:
        process_pgn_file(pgn_path, tensors_folder_path, polyglot_reader)
