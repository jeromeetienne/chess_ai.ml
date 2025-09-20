from stockfish import Stockfish

stockfish = Stockfish(path="/Users/jetienne/Downloads/stockfish/stockfish-macos-m1-apple-silicon")
stockfish.set_skill_level(1)
stockfish.set_depth(10)
stockfish.set_elo_rating(1350)
stockfish.set_position(['e2e4', 'e7e6'])
best_move = stockfish.get_best_move()
print(best_move)
