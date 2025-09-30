#!/usr/bin/env python3

# stdlib imports
import os
import pickle

# local imports
from src.libs.chess_extra import ChessExtra

__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.abspath(os.path.join(__dirname__, "..", "output"))

if __name__ == "__main__":
    all_moves_uci = ChessExtra.all_unique_moves()

    # sort the moves
    all_moves_uci = sorted(all_moves_uci)

    # Create uci2class mapping
    uci2class = {move: idx for idx, move in enumerate(all_moves_uci)}
    
    # Save the mapping to a file 
    file_path = os.path.join(data_folder_path, "uci_to_classindex.pickle")
    with open(file_path, "wb") as file_writer:
        pickle.dump(uci2class, file_writer)

    print(f"Saved {len(uci2class)} unique moves to {file_path}")

    
