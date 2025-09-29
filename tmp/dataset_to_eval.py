# stdlib imports
from asyncio import sleep
import os
import sys

# pip imports
import torch
import chess
from stockfish import Stockfish

# local imports
from src.libs.chess_extra import ChessExtra
from src.libs.encoding import Encoding

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_path = f"{__dirname__}/../output"

# Load dataset boards_tensor
boards_path = f"{output_path}/dataset_boards.pt"
boards_tensor = torch.load(boards_path)


# keep only the first 100 boards for testing
# boards_tensor = boards_tensor[1:100]

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
stockfish.set_depth(10)

for tensor_index, board_tensor in enumerate(boards_tensor):
    print(f'-'*40)
    print(f"Evaluating board {tensor_index+1}/{tensor_count} ...")
    # board reconstruction from tensor
    board = Encoding.board_from_tensor(board_tensor)

    # print(f'board from tensor:\n{board}')
    # print(ChessExtra.board_to_string(board, flip_board=False))


    board.set_castling_fen('-')  # allow no castling rights
    # TODO
    # - BUG BUG BUG in dataset
    # - boards_tensor does not encode turn (white/black to move)
    #   - it is always white to move in the tensor 
    # - boards_tensor does not encode castling rights
    # - boards_tensor does not encode en-passant rights
    # - boards_tensor does not encode 50-move rule counter
    # - boards_tensor does not encode repetition rule
    # - Q. how to encode that in the tensor?
    #   - https://github.com/iamlucaswolf/gym-chess/blob/master/gym_chess/alphazero/board_encoding.py <- find the code that encodes the board to tensor



    # check if the board is valid position - stockfish cannot evaluate invalid positions and crash on it
    if board.is_valid() == False:
        print(f'board status: {board.status()}  ')
        print(f"Panic Invalid board position:\n{board}")
        print(f"FEN: {board.fen()}")
        raise ValueError("Reconstructed board is not valid")

    # invalid board positions examples: no black king
    # r n . . k b . r
    # p p . . p p p p
    # . . p . . n . .
    # q . . . . . . b
    # . . B P . . . .
    # . . N . . N . P
    # P P P . . P P .
    # R . B Q K . . R

    # set the board position in Stockfish and evaluate it
    board_fen = board.fen()

    stockfish.set_fen_position(board_fen)

    try:
        stockfish_eval = stockfish.get_evaluation()
        # {"type": "cp", "value": 20} for +0.20
        # {"type": "mate", "value": 3} for mate in 3 moves
    except Exception as e:
        raise RuntimeError(f"Error getting evaluation from Stockfish: {e}")


    # convert stockfish evaluation to a float
    if( stockfish_eval["type"] == "cp" ):
        # centipawn evaluation
        board_eval = stockfish_eval["value"] / 100.0
    elif( stockfish_eval["type"] == "mate" ):
        # mate in N moves evaluation
        # FIXME find a way to encode that
        board_eval = stockfish_eval["value"]
    else:
        assert False, f"Unknown evaluation type: {stockfish_eval['type']}"

    print(f"Board evaluation: {board_eval}")
    eval_tensor[tensor_index] = board_eval
