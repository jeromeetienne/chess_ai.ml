#!/usr/bin/env python3

# stdlib imports
import os

# pip imports
import argparse

# local imports
from src.utils.pgn_utils import PGNUtils

def game_count_in_pgn(file_path: str) -> tuple[int, list[int]]:
    """
    Count the number of games in a PGN file by counting occurrences of the '[Event' tag.
    """
    start_offsets = []
    start_marker = '[Event'
    game_count = 0
    with open(file_path, 'r', encoding='utf-8') as file_reader:
        for line in file_reader:
            if line.startswith(start_marker):
                game_count += 1
                start_offsets.append(file_reader.tell())
    return game_count, start_offsets

    return game_count

def split_pgn_file(pgn_file_path: str, dst_folder: str, max_games_per_file: int = 5000):
    """
    Split a large PGN file into smaller files, each containing up to max_games_per_file games.
    The output files are named as {original_basename}_{N}_of_{total}.pgn
    """

    game_count, start_offsets = game_count_in_pgn(pgn_file_path)
    if game_count == 0:
        print(f"No games found in {pgn_file_path}. Skipping.")
        return
    
    file_count = (game_count + max_games_per_file - 1) // max_games_per_file
    src_basename = pgn_file_path.replace('.pgn', '')

    for file_index in range(file_count):
        dst_basename = f"{src_basename}_{file_index + 1}_of_{file_count}.pgn"
        dst_path = os.path.join(dst_folder, os.path.basename(dst_basename))

        # determine byte offsets for the current split
        offset_start = start_offsets[file_index * max_games_per_file] if file_index * max_games_per_file < game_count else None
        offset_end = start_offsets[(file_index + 1) * max_games_per_file] if (file_index + 1) * max_games_per_file < game_count else None

        # read from source and write to destination
        src_file = open(pgn_file_path, 'r', encoding='utf-8')
        dst_file = open(dst_path, 'w', encoding='utf-8')
        # seek to start offset if specified
        if offset_start is not None:
            src_file.seek(offset_start)
        # read until offset_end or EOF
        if offset_end is not None:
            bytes_to_read = offset_end - (offset_start if offset_start is not None else 0)
            data = src_file.read(bytes_to_read)
        else:
            data = src_file.read()
        # write to destination file
        dst_file.write(data)

        # close files
        src_file.close()
        dst_file.close()

if __name__ == "__main__":
    # Parse command line arguments - pgn_splitter.py -mgp 200 *.pgn
    argParser = argparse.ArgumentParser(description="Split large PGN files into smaller ones.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argParser.add_argument("--max-games-per-file", "-mgp", type=int, default=5000, help="Maximum number of games per split file.")
    argParser.add_argument("pgn_files", type=str, nargs='+', help="Path(s) to the PGN file(s) to split.")
    args = argParser.parse_args()

    print(f"Splitting PGN files: {args.pgn_files} into smaller files with max {args.max_games_per_file} games each.")

    for pgn_file in args.pgn_files:
        split_pgn_file(pgn_file, args.max_games_per_file)
