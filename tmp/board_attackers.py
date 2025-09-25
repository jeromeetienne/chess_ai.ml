

import chess
import os

from src.libs.chess_extra import ChessExtra
from src.libs.pgn_utils import PGNUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
pgn_folder_path = f"{__dirname__}/../data/pgn"
pgn_file_path = f"{pgn_folder_path}/lichess_elite_2014-09.pgn"

games = PGNUtils.load_games_from_pgn(pgn_file_path)

print(f"Total games in PGN file: {len(games)}")


game = games[0]

board = game.board()
moves = list(game.mainline_moves())
for move_index in range(len(moves)):
    board.push(moves[move_index])
    if move_index == 15:
        break

print(ChessExtra.board_to_string(board))


# def board_attacked_count_compute(board: chess.Board, color: chess.Color) -> list[list[int]]:
#     """
#     For each square on the board, count how many pieces of the opposite color are attacking it.
#     Return a 2D array of shape (8, 8) with the counts.
#     """
#     # initialize the 8x8 array with zeros
#     attacked_squares = [[0 for _ in range(8)] for _ in range(8)]
#     for square in chess.SQUARES:
#         attackers_squareset = board.attackers(color, square)
#         attackers_list = list(attackers_squareset)
#         if len(attackers_list) == 0:
#             continue
#         # print(f"Square {chess.square_name(square)} is attacked by {len(attackers_list)} pieces: {[chess.square_name(sq) for sq in attackers_list]}")
#         rank = chess.square_rank(square)
#         file = chess.square_file(square)
#         attacked_squares[rank][file] += len(attackers_list)

#     return attacked_squares

# def board_attacked_count_print(attacked_squares: list[list[int]]):
#     """
#     Print the attacked_squares 2D array in a readable format.
#     """
#     print("Attacked squares (number of attackers):")
#     print("  +-----------------------+")
#     for rank in range(7, -1, -1):
#         line = f"{rank + 1}|"
#         for file in range(8):
#             line += f" {attacked_squares[rank][file]} "
#         line += f"|"
#         print(line)
#     print("  +-----------------------+")
#     print("   A  B  C  D  E  F  G  H")



other_color = chess.BLACK if board.turn == chess.WHITE else chess.WHITE

my_board_attacked_count = ChessExtra.board_attacked_count_compute(board, other_color)
print('my attacked squares:')
print(ChessExtra.board_square_count_to_string(my_board_attacked_count))

opponent_board_attacked_count = ChessExtra.board_attacked_count_compute(board, board.turn)
print('opponent attacked squares:')
print(ChessExtra.board_square_count_to_string(opponent_board_attacked_count))