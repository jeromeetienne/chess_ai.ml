import pytest
import chess
import torch

from src.encoding.move_encoding_alphazero import MoveEncodingAlphaZero


def test_encode_decode_roundtrip_simple_moves():
    board = chess.Board()

    # pawn two-step
    move_1 = chess.Move.from_uci("e2e4")
    tensor_1 = MoveEncodingAlphaZero.encode_move_tensor(move_1, board.turn)
    assert tensor_1.shape == MoveEncodingAlphaZero.TENSOR_SHAPE
    move_decoded_1 = MoveEncodingAlphaZero.decode_move_tensor(tensor_1, board.turn)
    assert move_decoded_1 == move_1

    # knight move
    move_2 = chess.Move.from_uci("g1f3")
    tensor_2 = MoveEncodingAlphaZero.encode_move_tensor(move_2, board.turn)
    assert tensor_2.shape == MoveEncodingAlphaZero.TENSOR_SHAPE
    move_decoded_2 = MoveEncodingAlphaZero.decode_move_tensor(tensor_2, board.turn)
    assert move_decoded_2 == move_2

    # sliding move (after playing pawn to free path)
    board.push(move_1)
    board.push(chess.Move.from_uci("e7e5"))
    # bishop from f1 to c4 (requires clearing)
    board.push(chess.Move.from_uci("g1f3"))
    board.push(chess.Move.from_uci("b8c6"))
    move_3 = chess.Move.from_uci("f1c4")
    tensor_3 = MoveEncodingAlphaZero.encode_move_tensor(move_3, board.turn)
    assert tensor_3.shape == MoveEncodingAlphaZero.TENSOR_SHAPE
    move_decoded_3 = MoveEncodingAlphaZero.decode_move_tensor(tensor_3, board.turn)
    assert move_decoded_3 == move_3


def test_underpromotion_encoding_white_and_black():
    # Build positions where underpromotion is possible: white pawn on 7th rank
    board_w = chess.Board()
    # Clear board and set a white pawn on a7 (square a7 -> rank 6 file 0)
    board_w.clear()
    board_w.set_piece_at(chess.A7, chess.Piece(chess.PAWN, chess.WHITE))
    board_w.turn = chess.WHITE

    # underpromotion to knight on a8 (forward)
    move_kn = chess.Move.from_uci("a7a8n")
    tensor_kn = MoveEncodingAlphaZero.encode_move_tensor(move_kn, board_w.turn)
    move_decoded_kn = MoveEncodingAlphaZero.decode_move_tensor(tensor_kn, board_w.turn)
    assert move_decoded_kn == move_kn

    # underpromotion to rook on a8 (forward)
    move_ro = chess.Move.from_uci("a7a8r")
    tensor_ro = MoveEncodingAlphaZero.encode_move_tensor(move_ro, board_w.turn)
    move_decoded_ro = MoveEncodingAlphaZero.decode_move_tensor(tensor_ro, board_w.turn)
    assert move_decoded_ro == move_ro

    # black underpromotion (black pawn on second rank moving to first)
    board_b = chess.Board()
    board_b.clear()
    board_b.set_piece_at(chess.A2, chess.Piece(chess.PAWN, chess.BLACK))
    board_b.turn = chess.BLACK
    move_bq = chess.Move.from_uci("a2a1n")  # black promotes downwards, specify knight
    tensor_bq = MoveEncodingAlphaZero.encode_move_tensor(move_bq, board_b.turn)
    move_decoded_bq = MoveEncodingAlphaZero.decode_move_tensor(tensor_bq, board_b.turn)
    assert move_decoded_bq == move_bq


def test_queen_promotion_encoding_white_and_black():
    # White queen promotion (should be encoded as a sliding move, not underpromotion)
    board_w = chess.Board()
    board_w.clear()
    board_w.set_piece_at(chess.A7, chess.Piece(chess.PAWN, chess.WHITE))
    board_w.turn = chess.WHITE

    move_q = chess.Move.from_uci("a7a8q")
    tensor_q = MoveEncodingAlphaZero.encode_move_tensor(move_q, board_w.turn)
    move_decoded_q = MoveEncodingAlphaZero.decode_move_tensor(tensor_q, board_w.turn)
    assert move_decoded_q == move_q

    # Black queen promotion
    board_b = chess.Board()
    board_b.clear()
    board_b.set_piece_at(chess.H2, chess.Piece(chess.PAWN, chess.BLACK))
    board_b.turn = chess.BLACK

    move_qb = chess.Move.from_uci("h2h1q")
    tensor_qb = MoveEncodingAlphaZero.encode_move_tensor(move_qb, board_b.turn)
    move_decoded_qb = MoveEncodingAlphaZero.decode_move_tensor(tensor_qb, board_b.turn)
    assert move_decoded_qb == move_qb


def test_move_to_string_descriptions():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    desc = MoveEncodingAlphaZero.move_to_string(move, board.turn)
    assert "Slide" in desc and "from e2" in desc

    move_knight = chess.Move.from_uci("g1f3")
    desc_knight = MoveEncodingAlphaZero.move_to_string(move_knight, board.turn)
    assert "Knight jump" in desc_knight and "from g1" in desc_knight

    # Underpromotion
    board.clear()
    board.set_piece_at(chess.B7, chess.Piece(chess.PAWN, chess.WHITE))
    move_up = chess.Move.from_uci("b7b8n")
    desc_up = MoveEncodingAlphaZero.move_to_string(move_up, chess.WHITE)
    assert "Underpromotion" in desc_up and "to Knight" in desc_up


def test_encode_decode_all_knight_moves():
    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.D4, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.turn = chess.WHITE
    from_sq = chess.D4
    for dx, dy in MoveEncodingAlphaZero.KNIGHT_DIRS:
        fx, fy = divmod(from_sq, 8)
        tx, ty = fx + dx, fy + dy
        if 0 <= tx < 8 and 0 <= ty < 8:
            to_sq = tx * 8 + ty
            move = chess.Move(from_sq, to_sq)
            tensor = MoveEncodingAlphaZero.encode_move_tensor(move, board.turn)
            move_decoded = MoveEncodingAlphaZero.decode_move_tensor(tensor, board.turn)
            assert move_decoded == move


def test_encode_legal_moves_mask_matches_board():
    board = chess.Board()
    mask = MoveEncodingAlphaZero.encode_legal_moves_mask(board)
    # Count ones in mask should equal number of legal moves
    ones = int(mask.sum().item())
    assert ones == board.legal_moves.count()


def test_decode_invalid_tensor_shape_and_content_errors():
    # wrong shapemak
    bad = torch.zeros((72, 8, 8), dtype=torch.float32)
    with pytest.raises(ValueError):
        MoveEncodingAlphaZero.decode_move_tensor(bad, chess.WHITE)

    # multiple ones
    multi = torch.zeros(MoveEncodingAlphaZero.TENSOR_SHAPE, dtype=torch.float32)
    multi[0, 0, 0] = 1.0
    multi[1, 0, 0] = 1.0
    with pytest.raises(ValueError):
        MoveEncodingAlphaZero.decode_move_tensor(multi, chess.WHITE)
