import chess
import pytest

import torch

from src.encoding.move_encoding_uci2class import MoveEncodingUci2Class
from src.utils.uci2class_utils import Uci2ClassUtils


def round_trip(move_uci: str, color: chess.Color) -> str:
    """Helper: encode a UCI move to tensor and decode back to UCI string."""
    move = chess.Move.from_uci(move_uci)
    tensor = MoveEncodingUci2Class.encode_move_tensor(move, color)
    assert isinstance(tensor, torch.Tensor)
    decoded = MoveEncodingUci2Class.decode_move_tensor(tensor, color)
    return decoded.uci()


def test_basic_round_trip_white():
    # simple pawn step
    uci = "a2a3"
    out = round_trip(uci, chess.WHITE)
    assert out == uci


def test_basic_round_trip_black():
    # simple pawn step for black
    uci = "h7h6"
    out = round_trip(uci, chess.BLACK)
    assert out == uci


def test_promotion_moves_white():
    # promotions use suffixes like 'a7a8q' in the mapping
    for promo in ("q", "r", "b", "n"):
        uci = f"a7a8{promo}"
        out = round_trip(uci, chess.WHITE)
        assert out == uci


def test_promotion_moves_black():
    for promo in ("q", "r", "b", "n"):
        uci = f"h2h1{promo}"
        out = round_trip(uci, chess.BLACK)
        assert out == uci


def test_castling_moves():
    # castling notated as king moves in UCI: e1g1, e1c1 (white) and e8g8, e8c8 (black)
    for uci, color in [("e1g1", chess.WHITE), ("e1c1", chess.WHITE), ("e8g8", chess.BLACK), ("e8c8", chess.BLACK)]:
        out = round_trip(uci, color)
        assert out == uci


def test_move_index_bounds():
    # Ensure that class indices are within the number of classes
    num = Uci2ClassUtils.get_num_classes()
    class2uci = Uci2ClassUtils.get_class2uci(chess.WHITE)
    for idx in class2uci.keys():
        assert 0 <= idx < num


def test_invalid_move_raises_keyerror():
    # If a move is not present in the mapping, move_to_tensor should raise KeyError
    with pytest.raises(chess.InvalidMoveError):
        move = chess.Move.from_uci("zz1zz2")
        MoveEncodingUci2Class.encode_move_tensor(move, chess.WHITE)
