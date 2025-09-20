# stdlib imports
import os

# pip imports
import chess.pgn

__dirname__ = os.path.dirname(os.path.abspath(__file__))


def game_slice(src_game: chess.pgn.Game, move_index_start: int, move_index_end: int) -> chess.pgn.Game:
    src_mainline_moves = list(src_game.mainline_moves())
    # print(f"Total moves in first game: {len(src_mainline_moves)}")
    # print(f"first 5 moves: {[move.uci() for move in src_mainline_moves[:5]]}")

    # sanity check
    assert len(src_mainline_moves) > move_index_end, f"Game has only {len(src_mainline_moves)} moves, cannot extract moves {move_index_start} to {move_index_end}"
    assert move_index_start < move_index_end, f"move_index_start ({move_index_start}) must be less than move_index_end ({move_index_end})"

    # print(f"Total moves in first game: {len(src_mainline_moves)}")

    tmp_board = chess.Board()
    for move in src_mainline_moves[:move_index_start]:
        tmp_board.push_uci(move.uci())

    # create a new board
    tmp_board_fen = tmp_board.fen()
    dst_board = chess.Board(tmp_board_fen)

    for move in src_mainline_moves[move_index_start:move_index_end]:
        # print(f"applying move {move.uci()}")
        dst_board.push_uci(move.uci())

    dst_game = chess.pgn.Game.from_board(dst_board)
    return dst_game


###############################################################################
###############################################################################
#	read pgn file and display some info
###############################################################################
###############################################################################

file_path = f"{__dirname__}/../data/pgn/lichess_elite_2014-09.pgn"
games: list[chess.pgn.Game] = []
with open(file_path, 'r') as pgn_file:
    while True:
        game: chess.pgn.Game|None = chess.pgn.read_game(pgn_file)
        if game is None:
                break
        games.append(game)

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

move_index_start = 1
move_index_end = 3
src_game = games[0]

print_game(src_game)

dst_game = game_slice(src_game, move_index_start, move_index_end)

print_game(dst_game)

# print(f"FEN after applying moves {move_index_start} to {move_index_end}:")
# print(dst_game.board().unicode())
# for move in dst_game.mainline_moves():
#     print(f"move: {move.uci()}")
