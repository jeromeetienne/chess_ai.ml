#!/usr/bin/env python3

# stdlib imports
import os
import asyncio

# pip imports
import torch
from stockfish import Stockfish
from tqdm import tqdm
import chess.engine


# local imports
from src.libs.encoding import Encoding

async def main():
    __dirname__ = os.path.dirname(os.path.abspath(__file__))
    output_path = f"{__dirname__}/../output"

    basenames = os.listdir(output_path)
    # keep only the basename starting with dataset_boards
    basenames = [basename for basename in basenames if basename.startswith("dataset_boards")]

    for basename in basenames:
        # Load dataset boards_tensor
        boards_path = f"{output_path}/{basename}"
        boards_tensor = torch.load(boards_path)

        # # keep only the first boards for testing
        # boards_tensor = boards_tensor[:10000]

        tensor_count = boards_tensor.shape[0]
        print(f"Loaded {tensor_count} boards from {boards_path}")

        # create a tensor for the evaluation of each board
        eval_tensor = torch.zeros((tensor_count, ), dtype=torch.float32)

        ###############################################################################
        #   Board/Move reconstruction from tensor
        #

        # initialize stockfish
        stockfish_path = "/Users/jetienne/Downloads/stockfish/stockfish-macos-m1-apple-silicon"  # Update this path to your Stockfish binary
        stockfish = Stockfish(path=stockfish_path)
        # stockfish._set_option("Threads", 15)
        stockfish.set_depth(10)
        print(f"Stockfish version {stockfish.get_stockfish_major_version()}")

        # go thru all the boards in the tensor
        for tensor_index, board_tensor in tqdm(enumerate(boards_tensor), total=tensor_count, ncols=80, desc="Evaluating", unit="eval"):
            # board reconstruction from tensor
            board = Encoding.board_from_tensor(board_tensor)

            # set the board position in Stockfish and evaluate it
            stockfish.set_fen_position(board.fen())

            # get evaluation from stockfish
            stockfish_eval = stockfish.get_evaluation()


            # NOTE: about evaluating the board
            # if cp then, no more than abs(eval) < 10.0
            # if mate then, encode as 10.0 + (1 / X) so
            max_abs_cp_eval = 10.0  # in centipawns
            max_mate_in_x_eval = 1

            # convert stockfish evaluation to a float
            if( stockfish_eval["type"] == "cp" ):
                # centipawn evaluation
                board_eval = stockfish_eval["value"] / 100.0
                if board_eval > max_abs_cp_eval:
                    board_eval = max_abs_cp_eval
                elif board_eval < -max_abs_cp_eval:
                    board_eval = -max_abs_cp_eval
            elif( stockfish_eval["type"] == "mate" ):
                # mate in N moves evaluation
                matInX = stockfish_eval["value"]
                if matInX >= 0:
                    board_eval = max_abs_cp_eval + (max_mate_in_x_eval / (matInX if matInX != 0 else 1))
                elif matInX < 0:
                    board_eval = -max_abs_cp_eval - (max_mate_in_x_eval / (abs(matInX) if matInX != 0 else 1))
                else:
                    assert False, "Unreachable code"
            else:
                assert False, f"Unknown evaluation type: {stockfish_eval['type']}"

            # print(f"Board evaluation: {board_eval}")
            eval_tensor[tensor_index] = board_eval

        # save the eval tensor
        evals_path = boards_path.replace("dataset_boards", "dataset_evals")
        torch.save(eval_tensor, evals_path)
        print(f"Saved eval tensor to {evals_path}")

###############################################################################
#   Main entry point
#
if __name__ == "__main__":
    asyncio.run(main())