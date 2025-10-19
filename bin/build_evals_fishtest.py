#!/usr/bin/env python3

# stdlib imports
import os
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

    # iterate over all games
    for game in pgn_games:
        # iterate over all moves in the mainline, skipping positions in the opening book
        board = game.board()
        for node in game.mainline():
            # push the move to the board
            board.push(node.move)
            # skip if the position is in the opening book
            if skip_opening_book and polyglot_reader and ChessExtra.in_opening_book(board, polyglot_reader):
                continue

            # skip if is in endgame
            if skip_endgame and ChessExtra.is_endgame(board):
                continue

            # format of node.comment: '{evaluation_score}/{depth} {time}s'
            # e.g. '+0.18/21 2.7s' for eval +0.18 cp
            # e.g. '+M77/21 2.7s' for mate in 77

            # get evaluation score from node.comment
            comment_parts = node.comment.split("/", 1)
            assert len(comment_parts) == 2, f"Unexpected comment format: {node.comment}"
            score_str = comment_parts[0]

            # check if score_str is a float
            try:
                float(score_str)
                is_float = True
            except ValueError:
                is_float = False

            # convert score_str to chess.engine.Score
            if is_float:
                cp_value = int(float(score_str) * 100)
                chess_score = chess.engine.Cp(cp_value)
            elif score_str.startswith("+M") or score_str.startswith("-M"):
                mate_value = int(score_str[2:])
                chess_score = chess.engine.Mate(mate_value)
            else:
                raise ValueError(f"Unexpected eval_str format: {score_str}")

            # convert chess_score to a unified float score
            pgn_eval = unify_engine_score(chess_score)

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

    # process the pgn file
    pgn_evals = build_eval_array_from_pgn(pgn_path, polyglot_reader)

    # convert pgn_evals to a torch tensor
    evals_tensor = torch.tensor(pgn_evals, dtype=BoardEncoding.EVAL_DTYPE)

    # save evals_tensor to dst_path
    torch.save(evals_tensor, dst_path)

    # log the event
    print(f"Done")


###############################################################################
#   Check if output/pgn_splits contains fishtest pgn files
#
def folder_contains_fishtest_pgn(pgn_folder_path: str) -> bool:
    # sanity check - ensure output/pgn_splits contains fishtest pgn files - aka the basename is a 16 char hex string
    # - e.g. output/pgn_splits/5b07b25e0ebc5914abc12c6d.split_05_of_27.pgn
    # - e.g. output/pgn_splits/5b0974450ebc596ad1d1f757.split_27_of_28.pgn
    pgn_filenames = [f for f in os.listdir(pgn_folder_path) if f.endswith(".pgn")]
    if len(pgn_filenames) == 0:
        raise RuntimeError(f"No PGN files found in {pgn_folder_path}. Please ensure it contains fishtest pgn files.")
    contains_fishtest_pgn = False
    for pgn_filename in pgn_filenames:
        basename = pgn_filename.split(".")[0]
        if len(basename) >= 16 and all(c in "0123456789abcdef" for c in basename[:16]):
            contains_fishtest_pgn = True
            break
    return contains_fishtest_pgn


###############################################################################
#   Main Entry Point
#
if __name__ == "__main__":
    argParser = argparse.ArgumentParser(
        description="""Build evaluation tensors for chess positions fishtest games using Stockfish engine.
The evals are from the point of view of the player to move, and are the clamped centi-pawns.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argParser.add_argument("--max_files_count", "-fc", type=int, default=10, help="Maximum number of PGN files to process. 0 for no limit.")
    args = argParser.parse_args()
    # args = argParser.parse_args(['-fc', '4'])  # for testing only, remove this line for production

    # sanity check - ensure output/pgn_splits contains fishtest pgn files
    assert folder_contains_fishtest_pgn(
        pgn_folder_path
    ), f"output/pgn_splits does not contain fishtest pgn files. Please ensure it contains fishtest pgn files."

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
