# pip imports
import numpy as np
import torch
import chess

# local imports
from src.utils.uci2class_utils import Uci2ClassUtils


class BoardEncoding:

    BOARD_DTYPE = torch.int32
    MOVE_DTYPE = torch.int32  # class index as long
    EVAL_DTYPE = torch.float32  # evaluation as float

    # create a static property accesor for .OUTPUT_SHAPE
    @staticmethod
    def get_output_shape() -> tuple[int]:
        uci2class_white = Uci2ClassUtils.get_uci2class(chess.WHITE)
        num_classes = len(uci2class_white)
        return (num_classes,)

    @staticmethod
    def get_input_shape() -> tuple[int, int, int]:
        return (BoardEncoding.PLANE.PLANE_COUNT, 8, 8)

    class PLANE:
        ACTIVE_PAWN = 0
        ACTIVE_KNIGHT = 1
        ACTIVE_BISHOP = 2
        ACTIVE_ROOK = 3
        ACTIVE_QUEEN = 4
        ACTIVE_KING = 5
        OPPONENT_PAWN = 6
        OPPONENT_KNIGHT = 7
        OPPONENT_BISHOP = 8
        OPPONENT_ROOK = 9
        OPPONENT_QUEEN = 10
        OPPONENT_KING = 11

        REPETITION_2 = 12
        REPETITION_3 = 13

        TURN = 14
        ACTIVE_KINGSIDE_CASTLING_RIGHTS = 15
        ACTIVE_QUEENSIDE_CASTLING_RIGHTS = 16
        OPPONENT_KINGSIDE_CASTLING_RIGHTS = 17
        OPPONENT_QUEENSIDE_CASTLING_RIGHTS = 18

        HALFMOVE_CLOCK = 19
        FULLMOVE_NUMBER = 20

        # Total planes
        PLANE_COUNT = 21

    @staticmethod
    def board_to_tensor(board: chess.Board) -> torch.Tensor:
        # AlphaZero style board encoding
        # https://github.com/iamlucaswolf/gym-chess/blob/6a5eb43650c400a556505ec035cc3a3c5792f8b2/gym_chess/alphazero/board_encoding.py#L11C1-L33C1

        active_color = board.turn
        opponent_color = chess.WHITE if active_color == chess.BLACK else chess.BLACK

        # create an empty tensor - Use numpy first, it is 3-4 faster than torch for this
        board_numpy = np.zeros(BoardEncoding.get_input_shape(), dtype=np.uint16)

        ###############################################################################
        #   Piece planes
        #

        # Populate first 12 8x8 boards (where pieces are)
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            # flip rank and file if black to move
            rank = rank if active_color == chess.WHITE else 7 - rank
            file = file if active_color == chess.WHITE else 7 - file

            piece_color = 0 if piece.color == active_color else 6
            piece_type = piece.piece_type - 1
            board_numpy[piece_color + piece_type, rank, file] = 1

        ###############################################################################
        #   Repetition planes
        #

        # repetition_planes = np.zeros((repetition_plane_count, 8, 8))
        board_numpy[BoardEncoding.PLANE.REPETITION_2, :, :] = board.is_repetition(2)
        board_numpy[BoardEncoding.PLANE.REPETITION_3, :, :] = board.is_repetition(3)

        ###############################################################################
        #   Metadata planes
        #

        # Active player color
        board_numpy[BoardEncoding.PLANE.TURN, :, :] = int(board.turn)  # 1 for white, 0 for black

        # White player castling rights
        board_numpy[BoardEncoding.PLANE.ACTIVE_KINGSIDE_CASTLING_RIGHTS, :, :] = board.has_kingside_castling_rights(board.turn)
        board_numpy[BoardEncoding.PLANE.ACTIVE_QUEENSIDE_CASTLING_RIGHTS, :, :] = board.has_queenside_castling_rights(board.turn)

        # Black player castling rights
        board_numpy[BoardEncoding.PLANE.OPPONENT_KINGSIDE_CASTLING_RIGHTS, :, :] = board.has_kingside_castling_rights(opponent_color)
        board_numpy[BoardEncoding.PLANE.OPPONENT_QUEENSIDE_CASTLING_RIGHTS, :, :] = board.has_queenside_castling_rights(opponent_color)

        # Half-move clock
        board_numpy[BoardEncoding.PLANE.HALFMOVE_CLOCK, :, :] = board.halfmove_clock

        # Full move number
        board_numpy[BoardEncoding.PLANE.FULLMOVE_NUMBER, :, :] = board.fullmove_number

        ###############################################################################
        #   Return the board tensor
        #
        board_tensor = torch.tensor(board_numpy, dtype=BoardEncoding.BOARD_DTYPE)

        return board_tensor

    @staticmethod
    def board_from_tensor(board_tensor: torch.Tensor) -> chess.Board:
        """
        convert a board tensor of shape (21, 8, 8) to a chess.Board object
        """

        assert board_tensor.shape == BoardEncoding.get_input_shape(), f"board_tensor shape must be {BoardEncoding.get_input_shape()}, got {board_tensor.shape}"

        board = chess.Board()
        board.clear()  # clear the board

        active_color = bool(board_tensor[BoardEncoding.PLANE.TURN, 0, 0].item())
        opponent_color = not active_color

        ###############################################################################
        #   Parse Piece Planes and set pieces on the board
        #
        for board_idx in range(12):
            for rank in range(8):
                for file in range(8):
                    # skip empty squares
                    if board_tensor[board_idx, rank, file] == 0:
                        continue

                    # piece_count = 6 for unique pieces (P, N, B, R, Q, K)
                    piece_count = len(chess.PIECE_TYPES)
                    # compute square - flip file and rank if black to move
                    square = chess.square(file, rank) if active_color == chess.WHITE else chess.square(7 - file, 7 - rank)
                    # compute piece color
                    piece_color = active_color if board_idx < piece_count else opponent_color

                    # compute piece symbol index based on board index within the board_tensor
                    piece_symbol_idx = board_idx % piece_count + 1
                    # compute piece symbol
                    piece_symbol = chess.PIECE_SYMBOLS[piece_symbol_idx]
                    # piece symbol is uppercase for white, lowercase for black
                    piece_symbol = piece_symbol.upper() if piece_color == chess.WHITE else piece_symbol.lower()
                    # set piece on the board
                    piece = chess.Piece.from_symbol(piece_symbol)
                    board.set_piece_at(square, piece)

        ###############################################################################
        #   Parse Metadata Planes
        #

        # set turn
        board.turn = bool(board_tensor[BoardEncoding.PLANE.TURN, 0, 0].item())
        # set fullmove number
        board.fullmove_number = int(board_tensor[BoardEncoding.PLANE.FULLMOVE_NUMBER, 0, 0].item())
        # set castling rights
        castling_fen = ""
        if active_color == chess.WHITE:
            if board_tensor[BoardEncoding.PLANE.ACTIVE_KINGSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "K"
            if board_tensor[BoardEncoding.PLANE.ACTIVE_QUEENSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "Q"
            if board_tensor[BoardEncoding.PLANE.OPPONENT_KINGSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "k"
            if board_tensor[BoardEncoding.PLANE.OPPONENT_QUEENSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "q"
        else:
            if board_tensor[BoardEncoding.PLANE.OPPONENT_KINGSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "K"
            if board_tensor[BoardEncoding.PLANE.OPPONENT_QUEENSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "Q"
            if board_tensor[BoardEncoding.PLANE.ACTIVE_KINGSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "k"
            if board_tensor[BoardEncoding.PLANE.ACTIVE_QUEENSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "q"
        if castling_fen == "":
            castling_fen = "-"
        board.set_castling_fen(castling_fen)
        # set halfmove clock
        board.halfmove_clock = int(board_tensor[BoardEncoding.PLANE.HALFMOVE_CLOCK, 0, 0].item())

        assert board.is_valid(), "Reconstructed board is not valid"

        return board


if __name__ == "__main__":
    ###############################################################################
    #   Example usage
    #
    board = chess.Board()
    # board.push(chess.Move.from_uci("e2e4"))
    # move = chess.Move.from_uci("h7h6")

    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("c7c6"))

    print(f"Current board: {'white' if board.turn == chess.WHITE else 'black'}")

    board_tensor = BoardEncoding.board_to_tensor(board)

    reconstructed_board = BoardEncoding.board_from_tensor(board_tensor)
    print(f"Reconstructed board: {'white' if reconstructed_board.turn == chess.WHITE else 'black'}")
    print(reconstructed_board)
