# https://chatgpt.com/c/68e7d930-69ac-832b-8ae5-5fac5a399f6c

# stdlib imports
import math

# pip imports
import chess
import torch
import time


class MoveEncodingAlphaZero:
    """Container for move encoding ala AlphaZero.

    From "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"  https://arxiv.org/pdf/1712.01815:

    A move in chess may be described in two parts: selecting the piece to move, and then selecting among the legal moves for that piece.
    We represent the policy pi(a|s) by a 8x8x73 stack of planes encoding a probability distribution over 4,672 possible moves. Each of the 8x8
    positions identifies the square from which to “pick up” a piece. The first 56 planes encode possible ‘queen moves’ for any piece: a number
    of squares [1::7] in which the piece will be moved, along one of eight relative compass directions fN;NE;E; SE; S; SW;W;NWg. The next 8 planes
    encode possible knight moves for that piece. The final 9 planes encode possible underpromotions for pawn moves or captures in two possible
    diagonals, to knight, bishop or rook respectively. Other pawn moves or captures from the seventh rank are promoted to a queen.

    Chess Feature Planes: Total 73
    - Queen moves 56
    - Knight moves 8
    - Underpromotions 9

    NOTE: the tensor representation is a single integer tensor representing the class index.
    """

    # create a static property accesor for .OUTPUT_SHAPE
    @staticmethod
    def get_shape_tensor_output() -> tuple[int]:
        output_shape: tuple[int] = (MoveEncodingAlphaZero.CLASS_COUNT,)
        return output_shape

    @staticmethod
    def get_shape_tensor_classindex() -> tuple[int]:
        classindex_shape: tuple[int] = (1,)
        return classindex_shape

    # =============================================================================
    # Constants and encoding/decoding functions
    # =============================================================================
    MOVE_DTYPE = torch.int32  # class index as long
    """Tensor dtype for move encoding."""

    PLANE_COUNT = 73
    """Number of planes in the move encoding."""

    CLASS_COUNT = PLANE_COUNT * 8 * 8
    """Total number of move classes (73 planes x 64 from-squares)."""

    # TENSOR_SHAPE = (73, 8, 8)
    # """Shape of the move encoding tensor: (planes, ranks, files)."""

    SLIDING_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    """Directions for sliding pieces: N, S, E, W, NE, NW, SE, SW."""

    KNIGHT_DIRS = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
    """Knight move directions."""

    PROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    """Promotion piece types for underpromotions."""

    PROMOTION_DIRS = [(1, 0), (1, 1), (1, -1)]
    """Promotion directions: forward, forward-right, forward-left."""

    DIRECTION_NAMES = ["↑ (North)", "↓ (South)", "→ (East)", "← (West)", "↗ (North-East)", "↖ (North-West)", "↘ (South-East)", "↙ (South-West)"]
    """Human-readable names for sliding directions."""

    KNIGHT_NAMES = ["(+2, +1)", "(+2, -1)", "(-2, +1)", "(-2, -1)", "(+1, +2)", "(+1, -2)", "(-1, +2)", "(-1, -2)"]
    """Human-readable names for knight moves."""

    PROMO_PIECE_NAMES = {chess.KNIGHT: "Knight", chess.BISHOP: "Bishop", chess.ROOK: "Rook"}
    """Mapping from promotion piece type to name."""

    PROMO_DIR_NAMES = ["↑", "↗", "↖"]
    """Human-readable names for underpromotion directions."""

    @staticmethod
    def encode_move_tensor_classindex(move: chess.Move, color_to_move: chess.Color) -> torch.Tensor:
        """
        Return a tensor of shape (73, 8, 8) with 1 at move location and 0 elsewhere.
        """
        from_square, plane = MoveEncodingAlphaZero._move_to_square_plane(move, color_to_move)

        # compute class index
        class_index = plane * 64 + from_square
        # create tensor with single class_index
        move_tensor = torch.tensor(class_index, dtype=MoveEncodingAlphaZero.MOVE_DTYPE).reshape(MoveEncodingAlphaZero.get_shape_tensor_classindex())
        return move_tensor

    @staticmethod
    def decode_move_tensor_classindex(move_tensor: torch.Tensor, color_to_move: chess.Color) -> chess.Move:
        """
        Decode a tensor of shape (73*8*8,) back to a python-chess Move object.
        """
        if move_tensor.shape != MoveEncodingAlphaZero.get_shape_tensor_classindex():
            raise ValueError("Invalid tensor shape")

        # extract the single class index
        class_index = int(move_tensor.item())

        if class_index < 0 or class_index >= MoveEncodingAlphaZero.CLASS_COUNT:
            raise ValueError("Invalid class index in tensor")

        # derive from_square and plane from class_index
        plane = class_index // 64
        from_square = class_index % 64
        # decode move from from_square and plane
        move = MoveEncodingAlphaZero._move_from_square_plane(from_square, plane, color_to_move)

        return move

    # =============================================================================
    # Internal function
    # =============================================================================

    @staticmethod
    def _move_to_square_plane(move: chess.Move, color_to_move: chess.Color) -> tuple[int, int]:
        """
        Encode a python-chess Move object into (from_square, plane).

        Returns:
            from_square (int): The square index (0-63) from which the piece is moved.
            plane (int): The encoding plane index (0-72) representing the move type.

        Raises:
            ValueError: If the move cannot be encoded.
        """
        from_square = move.from_square
        to_square = move.to_square

        fx, fy = divmod(from_square, 8)
        tx, ty = divmod(to_square, 8)

        dx = tx - fx
        dy = ty - fy

        # 1. Underpromotion (must be checked before sliding moves so underpromotions
        #    to N/B/R are encoded in the last 9 planes rather than as a slide)
        if move.promotion in MoveEncodingAlphaZero.PROMOTION_PIECES:
            px_dir = 1 if color_to_move == chess.WHITE else -1
            rel_dx = tx - fx
            rel_dy = ty - fy
            for dir_idx, (px, py) in enumerate(MoveEncodingAlphaZero.PROMOTION_DIRS):
                expected_dx = px * px_dir
                expected_dy = py
                if rel_dx == expected_dx and rel_dy == expected_dy:
                    promo_type_index = MoveEncodingAlphaZero.PROMOTION_PIECES.index(move.promotion)
                    plane = 64 + dir_idx * 3 + promo_type_index  # 64–72
                    return from_square, plane

        # 2. Sliding moves
        for dir_index, (sx, sy) in enumerate(MoveEncodingAlphaZero.SLIDING_DIRS):
            for distance in range(1, 8):
                if (dx == sx * distance) and (dy == sy * distance):
                    plane = dir_index * 7 + (distance - 1)  # 0–55
                    return from_square, plane

        # 3. Knight moves
        for knight_index, (kx, ky) in enumerate(MoveEncodingAlphaZero.KNIGHT_DIRS):
            if (dx == kx) and (dy == ky):
                plane = 56 + knight_index  # 56–63
                return from_square, plane

        # 3. Underpromotion
        if move.promotion in MoveEncodingAlphaZero.PROMOTION_PIECES:
            px_dir = 1 if color_to_move == chess.WHITE else -1
            rel_dx = tx - fx
            rel_dy = ty - fy
            if rel_dx == px_dir:
                for dir_idx, (px, py) in enumerate(MoveEncodingAlphaZero.PROMOTION_DIRS):
                    if px == rel_dx and py == rel_dy:
                        promo_type_index = MoveEncodingAlphaZero.PROMOTION_PIECES.index(move.promotion)
                        plane = 64 + dir_idx * 3 + promo_type_index  # 64–72
                        return from_square, plane

        # throw an exception if move is illegal or cannot be encoded
        raise ValueError(f"Move {move.uci()} cannot be encoded")

    @staticmethod
    def _move_from_square_plane(from_square: int, plane: int, color_to_move: chess.Color) -> chess.Move:
        """
        Decode a (from_square, plane) back to a python-chess Move object.

        Arguments:
            from_square (int): The square index (0-63) from which the piece is moved.
            plane (int): The encoding plane index (0-72) representing the move type

        Returns:
            chess.Move: The decoded move.

        Raises:
            ValueError: If the move cannot be decoded.
        """
        fx, fy = divmod(from_square, 8)

        # --- 1. Sliding moves ---
        if 0 <= plane <= 55:
            dir_index = plane // 7
            distance = (plane % 7) + 1
            sx, sy = MoveEncodingAlphaZero.SLIDING_DIRS[dir_index]
            tx = fx + sx * distance
            ty = fy + sy * distance

            if 0 <= tx < 8 and 0 <= ty < 8:
                to_sq = tx * 8 + ty
                # If this sliding move reaches the final rank for the side to move,
                # interpret it as a queen promotion (AlphaZero encoding uses sliding
                # plane for queen promotions). We don't have board context here,
                # so assume promotions when moving to the last rank.
                if (color_to_move == chess.WHITE and tx == 7) or (color_to_move == chess.BLACK and tx == 0):
                    return chess.Move(from_square, to_sq, promotion=chess.QUEEN)
                return chess.Move(from_square, to_sq)

        # --- 2. Knight moves ---
        if 56 <= plane <= 63:
            knight_index = plane - 56
            kx, ky = MoveEncodingAlphaZero.KNIGHT_DIRS[knight_index]
            tx = fx + kx
            ty = fy + ky

            if 0 <= tx < 8 and 0 <= ty < 8:
                to_sq = tx * 8 + ty
                return chess.Move(from_square, to_sq)

        # --- 3. Underpromotion ---
        if 64 <= plane <= 72:
            px_dir = 1 if color_to_move == chess.WHITE else -1
            promo_plane = plane - 64
            dir_index = promo_plane // 3  # 0,1,2 for (↑, ↗, ↖)
            promo_index = promo_plane % 3  # 0 = knight, 1 = bishop, 2 = rook

            sx, sy = MoveEncodingAlphaZero.PROMOTION_DIRS[dir_index]
            tx = fx + sx * px_dir  # Only forward by 1
            ty = fy + sy

            if 0 <= tx < 8 and 0 <= ty < 8:
                to_sq = tx * 8 + ty
                promo_piece = MoveEncodingAlphaZero.PROMOTION_PIECES[promo_index]
                return chess.Move(from_square, to_sq, promotion=promo_piece)

        # raise an exception if move is illegal or cannot be decoded
        raise ValueError(f"Move from_square={from_square}, plane={plane} cannot be decoded")

    # =============================================================================
    # Auxiliary functions
    # =============================================================================

    @staticmethod
    def move_to_string(move: chess.Move, color_to_move: chess.Color) -> str:
        """
        Return a description of the move's encoding plane.
        """
        from_square, plane = MoveEncodingAlphaZero._move_to_square_plane(move, color_to_move)

        fx, fy = divmod(from_square, 8)
        from_name = chess.square_name(from_square)

        # Sliding moves
        if 0 <= plane <= 55:
            dir_index = plane // 7
            distance = (plane % 7) + 1
            direction = MoveEncodingAlphaZero.DIRECTION_NAMES[dir_index]
            output_str = f"Slide {direction} by {distance} from {from_name}"

        # Knight moves:
        elif 56 <= plane <= 63:
            knight_index = plane - 56
            delta_desc = MoveEncodingAlphaZero.KNIGHT_NAMES[knight_index]
            output_str = f"Knight jump {delta_desc} from {from_name}"

        # Underpromotion:
        elif 64 <= plane <= 72:
            promo_plane = plane - 64
            dir_index = promo_plane // 3
            promo_index = promo_plane % 3
            direction_name = MoveEncodingAlphaZero.PROMO_DIR_NAMES[dir_index]
            promo_piece = MoveEncodingAlphaZero.PROMO_PIECE_NAMES[MoveEncodingAlphaZero.PROMOTION_PIECES[promo_index]]
            output_str = f"Underpromotion {direction_name} to {promo_piece} from {from_name}"

        else:
            output_str = "Unknown plane type"

        return output_str

    # @staticmethod
    # def encode_legal_moves_mask(board: chess.Board) -> torch.Tensor:
    #     """
    #     Return a tensor (73, 8, 8) marking all legal moves with 1.0 at their encoded planes.
    #     Useful for masking policy logits.
    #     """
    #     move_mask_tensor = torch.zeros(MoveEncodingAlphaZero.TENSOR_SHAPE, dtype=torch.float32)
    #     for move in board.legal_moves:
    #         from_square, plane = MoveEncodingAlphaZero._move_to_square_plane(move, board.turn)
    #         rank, file = divmod(from_square, 8)
    #         move_mask_tensor[plane, rank, file] = 1.0
    #     return move_mask_tensor


# =============================================================================
# Main entry for quick testing
# =============================================================================
if __name__ == "__main__":
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    # =============================================================================
    # legal move mask
    # =============================================================================
    move_tensor = MoveEncodingAlphaZero.encode_move_tensor_classindex(move, board.turn)
    # legal_mask_tensor = MoveEncodingAlphaZero.encode_legal_moves_mask(board)

    print("Single move encoding (non-zero count):", move_tensor.sum().item())
    # print("Legal move mask count:", legal_mask_tensor.sum().item())
    print("Legal move count:", board.legal_moves.count())

    # Encode
    from_square, plane = MoveEncodingAlphaZero._move_to_square_plane(move, board.turn)
    print(f"Encoded: from_square={from_square}, plane={plane}")

    # Decode
    decoded_move = MoveEncodingAlphaZero._move_from_square_plane(from_square, plane, board.turn)
    assert decoded_move != None, "Decoding failed"
    print("Decoded move:", decoded_move, "→", decoded_move.uci())

    explanation = MoveEncodingAlphaZero.move_to_string(decoded_move, board.turn)
    print("Explanation:", explanation)

    # =============================================================================
    # benchmark .encode_move_tensor and .decode_move_tensor
    # =============================================================================

    bench_iterations = 100000

    # Benchmark .encode_move_tensor
    start_time = time.perf_counter()
    for _ in range(bench_iterations):
        MoveEncodingAlphaZero.encode_move_tensor_classindex(move, board.turn)
    end_time = time.perf_counter()
    print(f"Encoding time: {bench_iterations/(end_time - start_time):.0f} time per seconds")

    # Benchmark .decode_move_tensor
    start_time = time.perf_counter()
    for _ in range(bench_iterations):
        MoveEncodingAlphaZero.decode_move_tensor_classindex(move_tensor, board.turn)
    end_time = time.perf_counter()
    print(f"Decoding time: {bench_iterations/(end_time - start_time):.0f} time per seconds")
