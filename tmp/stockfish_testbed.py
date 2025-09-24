from stockfish import Stockfish

stockfish_path = "/Users/jetienne/Downloads/stockfish/stockfish-macos-m1-apple-silicon"  # Update this path to your Stockfish binary

stockfish = Stockfish(path=stockfish_path)
# Set the current board position in Stockfish
# stockfish.set_fen_position(board.fen())

# Get the best move from Stockfish
best_move = stockfish.get_best_move()

print(f"Stockfish suggests the move: {best_move}")

# evaluate the position
evaluation = stockfish.get_evaluation()
print(f"Stockfish evaluation: {evaluation}")


