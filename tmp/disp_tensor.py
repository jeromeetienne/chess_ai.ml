# stdlib imports
import os

# pip imports
import chess
import torch
import chess.polyglot

# local imports
from src.utils.pgn_utils import PGNUtils
from src.libs.chess_extra import ChessExtra


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "..", "data")
tensor_folder_path = os.path.join(data_folder_path, "pgn_tensors")


if __name__ == "__main__":
    # Load polyglot opening book
    polyglot_path = os.path.join(data_folder_path, "polyglot/lichess_pro_books/lpb-allbook.bin")
    polyglot_reader = chess.polyglot.open_reader(polyglot_path)

    # Get the first pgn file 
    pgn_paths = PGNUtils.get_pgn_paths()
    pgn_path = pgn_paths[0]

    # parse the pgn file
    pgn_games = PGNUtils.parse_pgn_file(pgn_path)

    # Go thru all the moves of the game, and store the board and move if the position is not in the opening book
    pgn_boards: list[chess.Board] = []
    pgn_moves: list[chess.Move] = []
    for game in pgn_games:
        board = chess.Board()
        for move in game.mainline_moves():
            board.push(move)
            # skip if the position is in the opening book
            if polyglot_reader and ChessExtra.is_in_opening_book(board, polyglot_reader):
                continue
            # append the board in pgn_boards
            pgn_boards.append(board.copy())
            # append the move in pgn_moves
            move_copy = chess.Move(move.from_square, move.to_square, promotion=move.promotion)
            pgn_moves.append(move_copy)



    boards_path = os.path.join(tensor_folder_path, os.path.basename(pgn_path).replace(".pgn", "_boards_tensor.pt"))
    boards_tensor = torch.load(boards_path)
    moves_path = os.path.join(tensor_folder_path, os.path.basename(pgn_path).replace(".pgn", "_moves_tensor.pt"))
    moves_tensor = torch.load(moves_path)

    ###############################################################################
    #   Display some info
    #
    print(f"Boards tensor shape: {boards_tensor.shape}")
    print(f"Moves tensor shape: {moves_tensor.shape}")

    pgn_board = pgn_boards[0]
    pgn_move = pgn_moves[0]
    print(f'pgn board:')
    print(pgn_board)
    print(f'pgn move: {pgn_move.uci()}')

    board_tensor = boards_tensor[0]
    move_tensor = moves_tensor[0]
    print("First board tensor:")
    print(board_tensor)

    print("First move tensor:")
    print(move_tensor)