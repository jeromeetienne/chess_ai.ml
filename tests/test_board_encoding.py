import chess
import torch

from src.libs.encoding import Encoding


def test_input_shape():
    """Input shape must match the number of planes and 8x8 board."""
    assert Encoding.get_input_shape() == (Encoding.PLANE.PLANE_COUNT, 8, 8)


def test_starting_position_planes_and_metadata():
    board = chess.Board()  # standard starting position
    tensor = Encoding.board_to_tensor(board)

    # shape and dtype
    assert tuple(tensor.shape) == Encoding.get_input_shape()
    assert tensor.dtype == Encoding.BOARD_DTYPE

    # piece counts: each side has 8 pawns, 2 knights, 2 bishops, 2 rooks, 1 queen, 1 king
    # active (white) planes indices 0-5
    pawn_plane = tensor[Encoding.PLANE.ACTIVE_PAWN]
    assert int(pawn_plane.sum()) == 8

    knight_plane = tensor[Encoding.PLANE.ACTIVE_KNIGHT]
    assert int(knight_plane.sum()) == 2

    bishop_plane = tensor[Encoding.PLANE.ACTIVE_BISHOP]
    assert int(bishop_plane.sum()) == 2

    rook_plane = tensor[Encoding.PLANE.ACTIVE_ROOK]
    assert int(rook_plane.sum()) == 2

    queen_plane = tensor[Encoding.PLANE.ACTIVE_QUEEN]
    assert int(queen_plane.sum()) == 1

    king_plane = tensor[Encoding.PLANE.ACTIVE_KING]
    assert int(king_plane.sum()) == 1

    # opponent (black) planes indices 6-11
    opp_pawn_plane = tensor[Encoding.PLANE.OPPONENT_PAWN]
    assert int(opp_pawn_plane.sum()) == 8

    # metadata: white to move, fullmove_number=1, halfmove_clock=0
    assert int(tensor[Encoding.PLANE.TURN, 0, 0].item()) == 1
    assert int(tensor[Encoding.PLANE.FULLMOVE_NUMBER, 0, 0].item()) == 1
    assert int(tensor[Encoding.PLANE.HALFMOVE_CLOCK, 0, 0].item()) == 0


def test_roundtrip_reconstruction_white():
    board = chess.Board()
    tensor = Encoding.board_to_tensor(board)
    reconstructed = Encoding.board_from_tensor(tensor)
    # FEN strings should match (castling rights and move numbers included)
    assert reconstructed.board_fen() == board.board_fen()
    assert reconstructed.turn == board.turn
    assert reconstructed.castling_rights == board.castling_rights
    assert reconstructed.fullmove_number == board.fullmove_number
    assert reconstructed.halfmove_clock == board.halfmove_clock


def test_roundtrip_reconstruction_black_perspective():
    # Create a position where black is to move by making one white move
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    # now black to move
    tensor = Encoding.board_to_tensor(board)
    # Ensure the TURN plane encodes black as 0
    assert int(tensor[Encoding.PLANE.TURN, 0, 0].item()) == 0

    reconstructed = Encoding.board_from_tensor(tensor)
    assert reconstructed.board_fen() == board.board_fen()
    assert reconstructed.turn == board.turn
    assert reconstructed.fullmove_number == board.fullmove_number


def _random_board(num_moves: int = 20) -> chess.Board:
    """Create a pseudo-random legal board by playing num_moves random legal moves from the starting position."""
    import random

    board = chess.Board()
    moves = 0
    while moves < num_moves and not board.is_game_over():
        legal = list(board.legal_moves)
        if not legal:
            break
        mv = random.choice(legal)
        board.push(mv)
        moves += 1
    return board


def test_cycle_random_positions_once():
    # test several random positions for a single encode/decode cycle
    for _ in range(8):
        board = _random_board(num_moves=30)
        tensor = Encoding.board_to_tensor(board)
        reconstructed = Encoding.board_from_tensor(tensor)
        assert reconstructed.board_fen() == board.board_fen()


def test_cycle_idempotence_multiple_times():
    # Pick a random position and apply the encode/decode cycle multiple times
    board = _random_board(num_moves=40)
    original_fen = board.board_fen()
    for _ in range(5):
        tensor = Encoding.board_to_tensor(board)
        board = Encoding.board_from_tensor(tensor)
        assert board.board_fen() == original_fen
