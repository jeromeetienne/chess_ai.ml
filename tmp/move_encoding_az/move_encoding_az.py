# https://chatgpt.com/c/68e7d930-69ac-832b-8ae5-5fac5a399f6c

import chess
import torch


class MoveEncodingAz:
    """Container for move encoding constants and helper functions.

    All previous top-level constants and functions were moved into this
    class as static members so callers can instantiate or use the class
    methods directly.
    """

    SLIDING_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    KNIGHT_DIRS = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
    PROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    PROMOTION_DIRS = [(1, 0), (1, 1), (1, -1)]  # Forward, forward-right, forward-left
    DIRECTION_NAMES = ["↑ (North)", "↓ (South)", "→ (East)", "← (West)", "↗ (North-East)", "↖ (North-West)", "↘ (South-East)", "↙ (South-West)"]
    KNIGHT_NAMES = ["(+2, +1)", "(+2, -1)", "(-2, +1)", "(-2, -1)", "(+1, +2)", "(+1, -2)", "(-1, +2)", "(-1, -2)"]
    PROMO_PIECE_NAMES = {chess.KNIGHT: "Knight", chess.BISHOP: "Bishop", chess.ROOK: "Rook"}
    PROMO_DIR_NAMES = ["↑", "↗", "↖"]

    @staticmethod
    def describe_plane(from_sq: int, plane: int, board: chess.Board):
        """
        Return (move, description_str). If not valid, returns (None, message).
        """
        decoded_move = MoveEncodingAz.decode_move_from_plane(from_sq, plane, board)

        if decoded_move is None:
            return None, f"Plane {plane}: INVALID / out of board"

        fx, fy = divmod(from_sq, 8)
        from_name = chess.square_name(from_sq)

        # Sliding moves
        if 0 <= plane <= 55:
            dir_index = plane // 7
            distance = (plane % 7) + 1
            direction = MoveEncodingAz.DIRECTION_NAMES[dir_index]
            desc = f"Slide {direction} by {distance} from {from_name}"

        # Knight moves:
        elif 56 <= plane <= 63:
            knight_index = plane - 56
            delta_desc = MoveEncodingAz.KNIGHT_NAMES[knight_index]
            desc = f"Knight jump {delta_desc} from {from_name}"

        # Underpromotion:
        elif 64 <= plane <= 72:
            promo_plane = plane - 64
            dir_index = promo_plane // 3
            promo_index = promo_plane % 3
            direction_name = MoveEncodingAz.PROMO_DIR_NAMES[dir_index]
            promo_piece = MoveEncodingAz.PROMO_PIECE_NAMES[MoveEncodingAz.PROMOTION_PIECES[promo_index]]
            desc = f"Underpromotion {direction_name} to {promo_piece} from {from_name}"

        else:
            desc = "Unknown plane type"

        return decoded_move, desc

    @staticmethod
    def move_to_plane(move: chess.Move, board: chess.Board) -> tuple[int, int] | None:
        """Return (from_square, plane) or None."""
        from_sq = move.from_square
        to_sq = move.to_square

        fx, fy = divmod(from_sq, 8)
        tx, ty = divmod(to_sq, 8)

        dx = tx - fx
        dy = ty - fy

        # 1. Sliding moves
        for dir_index, (sx, sy) in enumerate(MoveEncodingAz.SLIDING_DIRS):
            for distance in range(1, 8):
                if (dx == sx * distance) and (dy == sy * distance):
                    plane = dir_index * 7 + (distance - 1)  # 0–55
                    return from_sq, plane

        # 2. Knight moves
        for knight_index, (kx, ky) in enumerate(MoveEncodingAz.KNIGHT_DIRS):
            if (dx == kx) and (dy == ky):
                plane = 56 + knight_index  # 56–63
                return from_sq, plane

        # 3. Underpromotion
        if move.promotion in MoveEncodingAz.PROMOTION_PIECES:
            px_dir = 1 if board.turn == chess.WHITE else -1
            rel_dx = tx - fx
            rel_dy = ty - fy
            if rel_dx == px_dir:
                for dir_idx, (px, py) in enumerate(MoveEncodingAz.PROMOTION_DIRS):
                    if px == rel_dx and py == rel_dy:
                        promo_type_index = MoveEncodingAz.PROMOTION_PIECES.index(move.promotion)
                        plane = 64 + dir_idx * 3 + promo_type_index  # 64–72
                        return from_sq, plane

        return None  # Should not happen for proper legal moves

    @staticmethod
    def encode_move_tensor(move: chess.Move, board: chess.Board) -> torch.Tensor:
        """
        Return a tensor of shape (73, 8, 8) with 1 at move location and 0 elsewhere.
        """
        tensor = torch.zeros((73, 8, 8), dtype=torch.float32)
        result = MoveEncodingAz.move_to_plane(move, board)
        if result is None:
            return tensor  # illegal or unencodable move → return zero tensor

        from_sq, plane = result
        rank, file = divmod(from_sq, 8)
        tensor[plane, rank, file] = 1.0
        return tensor

    @staticmethod
    def encode_legal_moves_mask(board: chess.Board) -> torch.Tensor:
        """
        Return a tensor (73, 8, 8) marking all legal moves with 1.0 at their encoded planes.
        Useful for masking policy logits.
        """
        tensor = torch.zeros((73, 8, 8), dtype=torch.float32)
        for move in board.legal_moves:
            result = MoveEncodingAz.move_to_plane(move, board)
            if result is None:
                continue
            from_sq, plane = result
            rank, file = divmod(from_sq, 8)
            tensor[plane, rank, file] = 1.0
        return tensor

    @staticmethod
    def decode_move_from_plane(from_sq: int, plane: int, board: chess.Board):
        """
        Decode a (from_square, plane) back to a python-chess Move object.
        Returns None if decoding leads to an illegal move.
        """
        fx, fy = divmod(from_sq, 8)

        # --- 1. Sliding moves ---
        if 0 <= plane <= 55:
            dir_index = plane // 7
            distance = (plane % 7) + 1
            sx, sy = MoveEncodingAz.SLIDING_DIRS[dir_index]
            tx = fx + sx * distance
            ty = fy + sy * distance

            if 0 <= tx < 8 and 0 <= ty < 8:
                to_sq = tx * 8 + ty
                return chess.Move(from_sq, to_sq)

        # --- 2. Knight moves ---
        if 56 <= plane <= 63:
            knight_index = plane - 56
            kx, ky = MoveEncodingAz.KNIGHT_DIRS[knight_index]
            tx = fx + kx
            ty = fy + ky

            if 0 <= tx < 8 and 0 <= ty < 8:
                to_sq = tx * 8 + ty
                return chess.Move(from_sq, to_sq)

        # --- 3. Underpromotion ---
        if 64 <= plane <= 72:
            px_dir = 1 if board.turn == chess.WHITE else -1
            promo_plane = plane - 64
            dir_index = promo_plane // 3  # 0,1,2 for (↑, ↗, ↖)
            promo_index = promo_plane % 3  # 0 = knight, 1 = bishop, 2 = rook

            sx, sy = MoveEncodingAz.PROMOTION_DIRS[dir_index]
            tx = fx + sx * px_dir  # Only forward by 1
            ty = fy + sy

            if 0 <= tx < 8 and 0 <= ty < 8:
                to_sq = tx * 8 + ty
                promo_piece = MoveEncodingAz.PROMOTION_PIECES[promo_index]
                return chess.Move(from_sq, to_sq, promotion=promo_piece)

        return None  # Out of range or invalid


# =============================================================================
# Main entry for quick testing
# =============================================================================
if __name__ == "__main__":
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    single_move_tensor = MoveEncodingAz.encode_move_tensor(move, board)
    legal_mask = MoveEncodingAz.encode_legal_moves_mask(board)

    print("Single move encoding (non-zero count):", single_move_tensor.sum().item())
    print("Legal move mask count:", legal_mask.sum().item())

    # Encode
    result = MoveEncodingAz.move_to_plane(move, board)
    assert result is not None, "Encoding failed"
    fsq, plane = result
    print("Encoded:", fsq, plane)

    # Decode
    decoded = MoveEncodingAz.decode_move_from_plane(fsq, plane, board)
    assert decoded != None, "Decoding failed"
    print("Decoded:", decoded, "→", decoded.uci())

    decoded_move, explanation = MoveEncodingAz.describe_plane(fsq, plane, board)
    assert decoded_move is not None, "Description decoding failed"
    print("Move:", decoded_move.uci())
    print("Explanation:", explanation)
