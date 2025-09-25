#!/usr/bin/env python3

from src.libs.chess_extra import ChessExtra

if __name__ == "__main__":
    total_unique_moves = ChessExtra.all_unique_moves()
    print(f"All unique moves: {total_unique_moves}")
    print(f"Total unique moves: {len(total_unique_moves)}")
    print("error ? everybody say it is 1972. Where are the missing 4 !")