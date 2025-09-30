#!/usr/bin/env python3

# stdlib imports
import os
import json

# pip imports
import chess

# local imports
from src.libs.chess_extra import ChessExtra

__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.abspath(os.path.join(__dirname__, "..", "data"))

if __name__ == "__main__":
    all_moves_uci = ChessExtra.all_unique_moves()

    # sort the moves
    all_moves_uci = sorted(all_moves_uci)

    # Create uci2class mapping
    uci2class_white = [move_uci for move_uci in all_moves_uci]
    
    # Save the mapping to a file 
    white_file_path = os.path.join(data_folder_path, "uci2class_arr_white.json")
    with open(white_file_path, "w") as file_writer:
        json.dump(uci2class_white, file_writer, indent=4)
    print(f"Saved {len(uci2class_white)} unique moves to {white_file_path}")


    # Create uci2class mapping for black by mirroring XY the white moves
    uci2class_black = []
    for move_uci in uci2class_white:
        white_move = chess.Move.from_uci(move_uci)

        # Mirror the ranks and files for black
        black_from_square = chess.square(7 - chess.square_file(white_move.from_square), 7 - chess.square_rank(white_move.from_square))
        black_to_square = chess.square(7 - chess.square_file(white_move.to_square), 7 - chess.square_rank(white_move.to_square))
        black_move = chess.Move(from_square=black_from_square, to_square=black_to_square, promotion=white_move.promotion)

        black_move_uci = black_move.uci()
        uci2class_black.append(black_move_uci)

    # Save the mapping to a file 
    black_file_path = os.path.join(data_folder_path, "uci2class_arr_black.json")
    with open(black_file_path, "w") as file_writer:
        json.dump(uci2class_black, file_writer, indent=4)
    print(f"Saved {len(uci2class_black)} unique moves to {black_file_path}")    