import chess.polyglot
import chess.pgn
import os

__dirname__ = os.path.dirname(__file__)

pgn_path = os.path.join(__dirname__, "../data/pgn/5b46f00b0ebc5978f4be3ddb.pgn")

games: list[chess.pgn.Game] = []
with open(pgn_path, 'r') as pgn_file:
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        games.append(game)
        break


game = games[0]

# moves = list(game.mainline_moves())
# move = moves[0]
# print(f"move: {move}")

nodes = list(game.mainline())
node = nodes[0]
breakpoint()
print(f"node: {node}"   )