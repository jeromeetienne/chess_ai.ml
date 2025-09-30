# stdlib imports
import os

import chess.pgn

# local imports
from src.libs.chess_extra import ChessExtra
from src.utils.pgn_utils import PGNUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))



###############################################################################
###############################################################################
#	play e2e4
###############################################################################
###############################################################################

pgn_folder_path = f"{__dirname__}/../data/pgn"
pgn_file_paths = PGNUtils.all_pgn_file_paths(pgn_folder_path)

pgn_file_paths = pgn_file_paths[10:12]

# file_path = f"{__dirname__}/../data/pgn/lichess_elite_2014-09.pgn"
file_path = pgn_file_paths[0]
games = PGNUtils.load_games_from_pgn(file_path, 1000)

print(f"Loaded {len(games)} games from {file_path}")

# def game_has_been_completed(game: chess.pgn.Game) -> bool:
#     board = game.board()
#     for move in game.mainline_moves():
#         board.push(move)
#     return is_game_over

# count = 0
# avg_moves_count = 0
# for game_index, game in enumerate(games):
#     is_game_over = game_has_been_completed(game)
#     moves = list(game.mainline_moves())



#     print(f'Game {game_index+1}: {len(moves)} moves, is_game_over={is_game_over}')
#     avg_moves_count += len(moves)

# avg_moves_count /= len(games) if games else 1

# print(f"Number of games over: {count} / {len(games)}")
# print(f"Average moves per game: {avg_moves_count}")