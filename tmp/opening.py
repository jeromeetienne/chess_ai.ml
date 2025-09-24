import chess
import chess.polyglot
import os

__dirname__ = os.path.dirname(os.path.abspath(__file__))
polyglot_path = os.path.join(__dirname__, "../data/polyglot/gm2001.bin")  # Provide correct address to where your bin file is stored


polyglot_reader = chess.polyglot.open_reader(polyglot_path)
# n = sum(1 for _ in polyglot_reader.find_all(board))

# print(f"Number of entries: {len(opening_entries)}")
# print(opening_entries)

# for entry in opening_entries:
#     print(entry.move, entry.weight, entry.learn)


board = chess.Board()
while True:
    # Pick the entry at random based on weights
    try:
        opening_entry = polyglot_reader.weighted_choice(board)
    except IndexError:
        print("No opening entry found.")
        break

    print(f"opening {opening_entry}")

    print(board.unicode())
    print(f'move {board.fullmove_number}, {"White" if board.turn == chess.WHITE else "Black"} to play')
    print(f"Randomly selected entry: {opening_entry.move}, {opening_entry.weight}, {opening_entry.learn}")
    board.push(opening_entry.move)
