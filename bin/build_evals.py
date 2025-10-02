#!/usr/bin/env python3

# stdlib imports
import os
import asyncio
import typing
import time

# pip imports
import torch
import chess.engine
import psutil
import argparse


# local imports
from src.libs.encoding import Encoding

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_path = f"{__dirname__}/../output"
tensors_folder_path = os.path.join(__dirname__, "..", "data", "pgn_tensors")
stockfish_path = "/Users/jetienne/Downloads/stockfish/stockfish-macos-m1-apple-silicon"  # Update this path to your Stockfish binary


###############################################################################
#   Function to unify chess.engine.PovScore to a single numeric score
#
def unify_engine_score(pov_score: chess.engine.Score, mate_range: float = 100, max_centipawn: float = 300) -> float:
    """
    Convert chess.engine.PovScore to a single numeric score.
    - pov_score: chess.engine.PovScore object
    - mate_range: a large constant to scale mate scores

    Returns a float representing unified score. it is between [-max_centipawn-mate_range, max_centipawn+mate_range]
    """

    if pov_score.score() is not None:
        _score = pov_score.score()
        assert _score is not None
        # Just return centipawn score as is
        result = float(_score) / 100.0
        # Clamp to max_centipawn
        result = max(-max_centipawn, min(max_centipawn, result))
    elif pov_score.is_mate():
        # Convert mate in n moves to a large centipawn value with sign
        mate_moves = typing.cast(int, pov_score.mate())
        # # Defensive: avoid division by zero
        mate_moves = mate_moves if mate_moves != 0 else 1
        # Scale mate score
        result = (max_centipawn + mate_range / abs(mate_moves)) * (1 if mate_moves > 0 else -1)
    else:
        raise ValueError("Unknown PovScore type")

    return result


###############################################################################
#   Engine Pool class to manage multiple engine instances
#
class AnalyserEnginePool:
    """
    A pool of chess engines to analyze multiple boards concurrently.
    """

    def __init__(self, path: str, size: int = 10):
        self.path = path
        self.size = size
        self.engines = []
        self.lock = asyncio.Semaphore(size)

    async def start(self) -> None:
        for _ in range(self.size):
            transport, engine = await chess.engine.popen_uci(self.path)
            self.engines.append((transport, engine))

    async def board_to_povscore(self, boards_tensor: torch.Tensor, board_index: int, max_depth: int) -> chess.engine.Score:
        async with self.lock:
            # get an engine from the pool
            transport, engine = self.engines.pop()
            engine = typing.cast(chess.engine.UciProtocol, engine)

            # Log progress
            if board_index % 100 == 0:
                print(f".", end="", flush=True)

            # Analyze the board
            try:
                # get chess.Board from tensor
                board_tensor = boards_tensor[board_index]
                board = Encoding.board_from_tensor(board_tensor)
                # analyse the board
                info_dict = await engine.analyse(board, chess.engine.Limit(depth=max_depth), info=chess.engine.INFO_SCORE)
                # extract the pov score
                pov_score = info_dict.get("score")
                assert pov_score is not None, f"No score found in info: {info_dict}"
                # Get the relative score
                relative_score = pov_score.pov(board.turn)
                # print(f"Board {board_index}: {relative_score}")
            finally:
                self.engines.append((transport, engine))

            # return the relative score
            return relative_score

    async def close(self) -> None:
        # Close all the engine processes
        for transport, engine in self.engines:
            await engine.quit()
            transport.close()


###############################################################################
#   Main async function
#
async def main():
    eval_depth = 12

    ###############################################################################
    #   Start the pool of engines
    #

    # Get number of logical cores
    logical_cores = typing.cast(int, psutil.cpu_count(logical=True))

    # start the pool
    engine_count = logical_cores
    print(f"Starting a {engine_count} engine pool...", end=" ", flush=True)
    analyserPool = AnalyserEnginePool(stockfish_path, size=engine_count)
    await analyserPool.start()
    print("Done.")

    ###############################################################################
    boards_tensor_paths = os.listdir(tensors_folder_path)
    boards_tensor_paths.sort()
    boards_tensor_paths = [os.path.join(tensors_folder_path, basename) for basename in boards_tensor_paths if basename.endswith("_boards_tensor.pt")]
    if len(boards_tensor_paths) == 0:
        print(f"No boards tensor files found in {tensors_folder_path}")
        return

    ###############################################################################
    #   loop over all boards tensor files
    #
    eval_start_time = time.perf_counter()
    board_count_total = 0
    for boards_tensor_path in boards_tensor_paths:
        basename = os.path.basename(boards_tensor_path).replace("_boards_tensor.pt", "")

        evals_path = os.path.join(tensors_folder_path, f"{basename}_evals_tensor.pt")
        if os.path.exists(evals_path):
            print(f"{basename}.pgn already got a eval tensor, skipping.")
            continue

        # Load dataset boards_tensor
        boards_tensor = torch.load(boards_tensor_path)
        board_count = boards_tensor.shape[0]
        board_count_total += board_count

        print(f"{basename}.pgn processing {boards_tensor.shape[0]} boards", end=" ", flush=True)

        # get pov scores for all boards in the tensor
        tasks = [analyserPool.board_to_povscore(boards_tensor, board_index, max_depth=eval_depth) for board_index in range(board_count)]

        # Perform the evaluation tasks concurrently
        time_start = time.perf_counter()
        relative_scores: list[chess.engine.Score] = await asyncio.gather(*tasks)
        time_elapsed = time.perf_counter() - time_start

        print(f"Done.")
        print(f"{basename}.pgn evaluated in {time_elapsed:.2f} seconds, avg {board_count/time_elapsed:.2f} boards/sec")
        ###############################################################################
        #   Build and save eval tensor from the relative scores
        #

        # unify all the pov scores
        unified_scores: list[float] = [unify_engine_score(relative_score) for relative_score in relative_scores]

        # build eval tensor
        eval_tensor = torch.tensor(unified_scores, dtype=torch.float32)

        # Save eval tensor
        torch.save(eval_tensor, evals_path)
        # print(f"Saved eval tensor with {eval_tensor.shape[0]} evaluations to {evals_path}")

    eval_elapsed_time = time.perf_counter() - eval_start_time
    print(f"Evaluated {board_count_total} boards in {eval_elapsed_time} seconds, avg {board_count_total/eval_elapsed_time} boards/sec")

    # Close the pool of engines
    await analyserPool.close()


###############################################################################
#   Main entry point
#
if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="Build evaluation tensors for chess positions using Stockfish engine.")
    args = argParser.parse_args()

    asyncio.run(main())
