import numpy as np
import chess
import chess.pgn as pgn


def board_to_matrix(board: chess.Board) -> np.ndarray:
    # 8x8 is a size of the chess board.
    # 12 = number of unique pieces.
    # 13th board for legal moves (WHERE we can move)
    # maybe 14th for squares FROM WHICH we can move? idk
    matrix = np.zeros((14, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves board (13th 8x8 board)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    for move in legal_moves:
        from_square = move.from_square
        row_from, col_from = divmod(from_square, 8)
        matrix[13, row_from, col_from] = 1

    return matrix

def create_input_for_nn(games: list[pgn.Game]) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)


def encode_moves(moves: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int

def load_games_from_pgn(file_path: str) -> list[pgn.Game]:
    games: list[pgn.Game] = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game: pgn.Game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games
