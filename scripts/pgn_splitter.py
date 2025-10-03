#!/usr/bin/env python3

# stdlib imports
import os

# pip imports
import argparse
import gzip

# local imports
from src.utils.pgn_utils import PGNUtils

def game_count_in_pgn(file_path: str) -> tuple[int, list[int]]:
    """
    Count the number of games in a PGN file by counting occurrences of the '[Event' tag.
    """
    start_offsets = []
    start_marker = '[Event'
    game_count = 0

    file_reader = gzip.open(file_path, 'rt', encoding='utf-8') if file_path.endswith('.gz') else open(file_path, 'r', encoding='utf-8')
    with file_reader:
        current_offset = 0
        for line in file_reader:
            if line.startswith(start_marker):
                game_count += 1
                start_offsets.append(current_offset)
            current_offset += len(line)
    return game_count, start_offsets

    return game_count

def split_pgn_file(src_path: str, dst_folder: str, games_per_file: int = 500):
    """
    Split a large PGN file into smaller files, each containing up to games_per_file games.
    The output files are named as {original_basename}_{N}_of_{total}.pgn
    """

    game_count, start_offsets = game_count_in_pgn(src_path)
    if game_count == 0:
        print(f"No games found in {src_path}. Skipping.")
        return
    
    file_count = (game_count + games_per_file - 1) // games_per_file
    src_basename = os.path.basename(src_path).replace('.pgn.gz', '').replace('.pgn', '')

    # open source file (handle .gz if needed)
    src_file = gzip.open(src_path, 'rt', encoding='utf-8') if src_path.endswith('.pgn.gz') else open(src_path, 'r', encoding='utf-8')

    for file_index in range(file_count):
        dst_basename = f"{src_basename}.split_{str(file_index + 1).rjust(2,'0')}_of_{str(file_count).rjust(2,'0')}.pgn"
        dst_path = os.path.join(dst_folder, os.path.basename(dst_basename))

        # determine byte offsets for the current split
        offset_start = start_offsets[file_index * games_per_file] if file_index * games_per_file < game_count else None
        offset_end = start_offsets[(file_index + 1) * games_per_file] if (file_index + 1) * games_per_file < game_count else None

        # read from source and write to destination
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
        dst_file.close()
    
    # close source file
    src_file.close()

    return game_count

###############################################################################
#   main entry point
#
if __name__ == "__main__":
    # Parse command line arguments - pgn_splitter.py -mgp 200 *.pgn
    argParser = argparse.ArgumentParser(description="Split large PGN files into smaller ones.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argParser.add_argument("--games-per-file", "-gpf", type=int, default=200, help="Maximum number of games per split file.")
    argParser.add_argument("--dst-folder", "-d", type=str, default=None, help="Destination folder for split files (default: same as source file).")
    argParser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")
    argParser.add_argument("pgn_paths", type=str, nargs='+', help="Path(s) to the PGN file(s) to split.")
    args = argParser.parse_args()
    # args = argParser.parse_args(['./data/fishtest_pgns/18-05-25/5b07b25e0ebc5914abc12c6d/5b07b25e0ebc5914abc12c6d.pgn', '--max-games-per-file', '2000'])

    # print(f"Splitting PGN files: {args.pgn_paths} into smaller files with max {args.games_per_file} games each.")


    for pgn_path in args.pgn_paths:
        # ignore pgn_path which contain '.split_'
        if '.split_' in pgn_path:
            print(f"Skipping already split file: {pgn_path}")
            continue

        if args.verbose:
            print(f"Processing file: {pgn_path}", end='', flush=True)
        pgn_path = os.path.abspath(pgn_path)
        dst_folder = args.dst_folder if args.dst_folder is not None else os.path.dirname(pgn_path)
        game_count = split_pgn_file(pgn_path, dst_folder, args.games_per_file)

        if args.verbose:
            print(f" - done. {game_count} games found.")
