import pytest
import chess
import torch

from src.libs.move_encoding_az import MoveEncodingAz


def test_encode_decode_roundtrip_simple_moves():
    board = chess.Board()

    # pawn two-step
    move_1 = chess.Move.from_uci("e2e4")
    t1 = MoveEncodingAz.encode_move_tensor(move_1, board.turn)
    assert t1.shape == MoveEncodingAz.TENSOR_SHAPE
    move_decoded_1 = MoveEncodingAz.decode_move_tensor(t1, board.turn)
    assert move_decoded_1 == move_1

    # knight move
    move_2 = chess.Move.from_uci("g1f3")
    tensor_2 = MoveEncodingAz.encode_move_tensor(move_2, board.turn)
    assert tensor_2.shape == MoveEncodingAz.TENSOR_SHAPE
    move_decoded_2 = MoveEncodingAz.decode_move_tensor(tensor_2, board.turn)
    assert move_decoded_2 == move_2

    # sliding move (after playing pawn to free path)
    board.push(move_1)
    board.push(chess.Move.from_uci("e7e5"))
    # bishop from f1 to c4 (requires clearing)
    board.push(chess.Move.from_uci("g1f3"))
    board.push(chess.Move.from_uci("b8c6"))
    move_3 = chess.Move.from_uci("f1c4")
    tensor_3 = MoveEncodingAz.encode_move_tensor(move_3, board.turn)
    assert tensor_3.shape == MoveEncodingAz.TENSOR_SHAPE
    move_decoded_3 = MoveEncodingAz.decode_move_tensor(tensor_3, board.turn)
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
    t_kn = MoveEncodingAz.encode_move_tensor(move_kn, board_w.turn)
    move_decoded_kn = MoveEncodingAz.decode_move_tensor(t_kn, board_w.turn)
    assert move_decoded_kn == move_kn

    # underpromotion to rook on a8 (forward)
    move_ro = chess.Move.from_uci("a7a8r")
    t_ro = MoveEncodingAz.encode_move_tensor(move_ro, board_w.turn)
    move_decoded_ro = MoveEncodingAz.decode_move_tensor(t_ro, board_w.turn)
    assert move_decoded_ro == move_ro

    # black underpromotion (black pawn on second rank moving to first)
    board_b = chess.Board()
    board_b.clear()
    board_b.set_piece_at(chess.A2, chess.Piece(chess.PAWN, chess.BLACK))
    board_b.turn = chess.BLACK
    move_bq = chess.Move.from_uci("a2a1n")  # black promotes downwards, specify knight
    t_bq = MoveEncodingAz.encode_move_tensor(move_bq, board_b.turn)
    move_decoded_bq = MoveEncodingAz.decode_move_tensor(t_bq, board_b.turn)
    assert move_decoded_bq == move_bq


def test_encode_legal_moves_mask_matches_board():
    board = chess.Board()
    mask = MoveEncodingAz.encode_legal_moves_mask(board)
    # Count ones in mask should equal number of legal moves
    ones = int(mask.sum().item())
    assert ones == board.legal_moves.count()


def test_decode_invalid_tensor_shape_and_content_errors():
    # wrong shape
    bad = torch.zeros((72, 8, 8), dtype=torch.float32)
    with pytest.raises(ValueError):
        MoveEncodingAz.decode_move_tensor(bad, chess.WHITE)

    # multiple ones
    multi = torch.zeros(MoveEncodingAz.TENSOR_SHAPE, dtype=torch.float32)
    multi[0, 0, 0] = 1.0
    multi[1, 0, 0] = 1.0
    with pytest.raises(ValueError):
        MoveEncodingAz.decode_move_tensor(multi, chess.WHITE)
