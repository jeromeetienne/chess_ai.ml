# stdlib imports
import os
import sys

# pip imports
import torch
import chess

# local imports
from src.libs.io_utils import IOUtils
from src.libs.pgn_utils import PGNUtils
from src.libs.encoding_utils import EncodingUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_path = f"{__dirname__}/../output"


boards_path = f"{output_path}/dataset_boards.pt"
boards_tensor = torch.load(boards_path)

moves_path = f"{output_path}/dataset_moves.pt"
moves_tensor = torch.load(moves_path)

uci_to_classindex = IOUtils.load_uci_to_classindex(folder_path=output_path)
classindex_to_uci = IOUtils.classindex_to_uci_inverse_mapping(uci_to_classindex)

###############################################################################
#   Board/Move reconstruction from tensor
#

position_idx = 14

board_tensor = boards_tensor[position_idx]
move_tensor = moves_tensor[position_idx]

# board reconstruction from tensor
board_recontruct = EncodingUtils.board_from_tensor(board_tensor)
print(f"Board reconstruction:\n{board_recontruct}")

# Move reconstruction from tensor
move_tensor_uci = EncodingUtils.move_from_tensor(move_tensor, classindex_to_uci)
print(f"Move reconstruction UCI: {move_tensor_uci}")

###############################################################################
#   PGN loading and board/move extraction to check against tensor reconstruction
#
pgn_folder_path = f"{__dirname__}/../data/pgn"
pgn_file_paths = PGNUtils.all_pgn_file_paths(pgn_folder_path)
games = PGNUtils.load_games_from_pgn(pgn_file_paths[0], 1000)

game = games[0]


print(f"Game:\n{game}")

board = game.board()
mainline_moves = list(game.mainline_moves())
move_chess_uci = mainline_moves[position_idx].uci()

# board.fullmove_number starts at 1 and increments after black's move
# board.halfmove_clock number of halfmoves since last pawn move or capture

board_chess = chess.Board()
for move in mainline_moves[:position_idx]:
    board_chess.push(move)

print('='*40)
print(f'board turn: {"white" if board_chess.turn == chess.WHITE else "black"}')
print(f"Board chess:\n{board_chess}")
print(f"Move chess UCI: {move_chess_uci}")
print(f'board chess valid: {board_chess.is_valid()}')
print(f'board fen: {board_chess.fen()}')