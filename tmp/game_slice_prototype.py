# stdlib imports
import os

# pip imports
import chess.pgn

# local imports
from src.libs.pgn_utils import PGNUtils
from src.libs.chess_extra import ChessExtra

__dirname__ = os.path.dirname(os.path.abspath(__file__))

###############################################################################
###############################################################################
#	read pgn file and display some info
###############################################################################
###############################################################################

file_path = f"{__dirname__}/../data/pgn/lichess_elite_2014-09.pgn"
games = PGNUtils.load_games_from_pgn(file_path)
print(f"Total games in PGN file: {len(games)}")

###############################################################################
###############################################################################
#	play e2e4
###############################################################################
###############################################################################

def print_game(game: chess.pgn.Game):
    print(game.board().unicode())
    for move in game.mainline_moves():
        print(f"move: {move.uci()}")

move_index_start = 3
move_index_end = 10
src_game = games[0]

print_game(src_game)

dst_game = ChessExtra.game_slice(src_game, move_index_start, move_index_end)

board = dst_game.board()
print(board.fen())

# print_game(dst_game)

# Display the pgn of the sliced game
print("Sliced game PGN:")
print(dst_game)