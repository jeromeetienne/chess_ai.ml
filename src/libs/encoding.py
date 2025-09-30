# pip imports
import numpy as np
import torch
import chess
import chess.pgn
import chess.polyglot
from tqdm import tqdm

# local imports
from src.libs.chess_extra import ChessExtra
from src.utils.uci2class_utils import Uci2ClassUtils


class Encoding:

    INPUT_SHAPE = (21, 8, 8)  # (channels, height, width)
    BOARD_DTYPE = torch.uint8
    MOVE_DTYPE = torch.int32  # class index as long

    class PLANE:
        WHITE_PAWN = 0
        WHITE_KNIGHT = 1
        WHITE_BISHOP = 2
        WHITE_ROOK = 3
        WHITE_QUEEN = 4
        WHITE_KING = 5
        BLACK_PAWN = 6
        BLACK_KNIGHT = 7
        BLACK_BISHOP = 8
        BLACK_ROOK = 9
        BLACK_QUEEN = 10
        BLACK_KING = 11

        REPETITION_2 = 12
        REPETITION_3 = 13

        TURN = 14
        FULLMOVE_NUMBER = 15
        WHITE_KINGSIDE_CASTLING_RIGHTS = 16
        WHITE_QUEENSIDE_CASTLING_RIGHTS = 17
        BLACK_KINGSIDE_CASTLING_RIGHTS = 18
        BLACK_QUEENSIDE_CASTLING_RIGHTS = 19
        NO_PROGRESS_COUNTER = 20

        # Total planes
        PLANE_COUNT = 21

    @staticmethod
    def board_to_tensor(board: chess.Board) -> torch.Tensor:
        # AlphaZero style board encoding
        # https://github.com/iamlucaswolf/gym-chess/blob/6a5eb43650c400a556505ec035cc3a3c5792f8b2/gym_chess/alphazero/board_encoding.py#L11C1-L33C1

        assert Encoding.INPUT_SHAPE == (21, 8, 8), f"INPUT_SHAPE must be (21, 8, 8), got {Encoding.INPUT_SHAPE}"

        # create an empty tensor - Use numpy first, it is 3-4 faster than torch for this
        board_numpy = np.zeros(Encoding.INPUT_SHAPE, dtype=np.uint8)

        ###############################################################################
        #   Piece planes
        #

        # Populate first 12 8x8 boards (where pieces are)
        for square, piece in board.piece_map().items():
            row, col = divmod(square, 8)
            piece_color = 0 if piece.color == chess.WHITE else 6
            piece_type = piece.piece_type - 1
            board_numpy[piece_color + piece_type, row, col] = 1

        ###############################################################################
        #   Repetition planes
        #

        # repetition_planes = np.zeros((repetition_plane_count, 8, 8))
        board_numpy[Encoding.PLANE.REPETITION_2, :, :] = board.is_repetition(2)
        board_numpy[Encoding.PLANE.REPETITION_3, :, :] = board.is_repetition(3)

        ###############################################################################
        #   Metadata planes
        #

        # Active player color
        board_numpy[Encoding.PLANE.TURN, :, :] = int(board.turn)  # 1 for white, 0 for black

        # Total move count
        board_numpy[Encoding.PLANE.FULLMOVE_NUMBER, :, :] = board.fullmove_number

        # White player castling rights
        board_numpy[Encoding.PLANE.WHITE_KINGSIDE_CASTLING_RIGHTS, :, :] = board.has_kingside_castling_rights(chess.WHITE)
        board_numpy[Encoding.PLANE.WHITE_QUEENSIDE_CASTLING_RIGHTS, :, :] = board.has_queenside_castling_rights(chess.WHITE)

        # Black player castling rights
        board_numpy[Encoding.PLANE.BLACK_KINGSIDE_CASTLING_RIGHTS, :, :] = board.has_kingside_castling_rights(chess.BLACK)
        board_numpy[Encoding.PLANE.BLACK_QUEENSIDE_CASTLING_RIGHTS, :, :] = board.has_queenside_castling_rights(chess.BLACK)

        # No-progress counter
        board_numpy[Encoding.PLANE.NO_PROGRESS_COUNTER, :, :] = board.halfmove_clock

        ###############################################################################
        #   Return the board tensor
        #
        board_tensor = torch.tensor(board_numpy, dtype=Encoding.BOARD_DTYPE)
        return board_tensor

    @staticmethod
    def board_from_tensor(board_tensor: torch.Tensor) -> chess.Board:
        """
        convert a board tensor of shape (21, 8, 8) to a chess.Board object
        """

        assert board_tensor.shape == Encoding.INPUT_SHAPE, f"board_tensor shape must be {Encoding.INPUT_SHAPE}, got {board_tensor.shape}"

        board = chess.Board()
        board.clear()  # clear the board
        ###############################################################################
        #   Parse Piece Planes and set pieces on the board
        #
        for board_idx in range(12):
            for row in range(8):
                for col in range(8):
                    if board_tensor[board_idx, row, col] == 0:
                        continue
                    # piece_count = 6 for unique pieces (P, N, B, R, Q, K)
                    piece_count = len(chess.PIECE_TYPES)
                    # compute square
                    square = chess.square(col, row)
                    # compute piece color
                    piece_color = chess.WHITE if board_idx < piece_count else chess.BLACK

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
        board.turn = bool(board_tensor[Encoding.PLANE.TURN, 0, 0].item())
        # set fullmove number
        board.fullmove_number = int(board_tensor[Encoding.PLANE.FULLMOVE_NUMBER, 0, 0].item())
        # set castling rights
        # TMP: disable castling rights for now
        board.set_castling_fen("-")
        # FIXME this seems fragile ... check this out
        # if board_tensor[Encoding.PLANE.BLACK_KINGSIDE_CASTLING_RIGHTS, 0, 0].item():
        #     if board.turn == chess.WHITE:
        #         board.castling_rights |= chess.BB_H1
        #     else:
        #         board.castling_rights |= chess.BB_A8
        # if board_tensor[15, 0, 0].item():
        #     if board.turn == chess.WHITE:
        #         board.castling_rights |= chess.BB_A1
        #     else:
        #         board.castling_rights |= chess.BB_H8
        # set halfmove clock
        board.halfmove_clock = int(board_tensor[19, 0, 0].item())

        assert board.is_valid(), "Reconstructed board is not valid"

        return board

    @staticmethod
    def board_tensor_flip(board_tensor: torch.Tensor, in_place: bool = False) -> torch.Tensor:
        """
        Flip the board tensor to switch perspective between white and black.
        This involves rotating the piece planes and swapping the piece colors.
        """

        assert board_tensor.shape == Encoding.INPUT_SHAPE, f"board_tensor shape must be {Encoding.INPUT_SHAPE}, got {board_tensor.shape}"

        # Rotate all planes by 180 degrees
        rotated_tensor = torch.flip(input=board_tensor, dims=[1, 2])

        if in_place == False:
            return rotated_tensor

        board_tensor.copy_(rotated_tensor)
        return board_tensor

    # @staticmethod
    # def move_to_tensor(move_uci: str, uci_to_classindex: dict[str, int]) -> torch.Tensor:
    #     """
    #     Converts a move in UCI string format to a scalar tensor representing its class index.

    #     Args:
    #         move_uci (str): The move in UCI (Universal Chess Interface) string format (e.g., 'e2e4').
    #         uci_to_classindex (dict[str, int]): A mapping from UCI move strings to their corresponding class indices.

    #     Returns:
    #         torch.Tensor: A scalar tensor containing the class index of the move, with dtype specified by Encoding.MOVE_DTYPE.

    #     Raises:
    #         KeyError: If the provided move_uci is not found in the uci_to_classindex mapping.
    #     """
    #     # convert move in UCI format to class index
    #     class_index = uci_to_classindex[move_uci]
    #     # convert class index to tensor
    #     # moves_tensor_dtype = EncodingUtils.__numpy_dtype_to_torch(EncodingUtils.MOVE_DTYPE)
    #     move_tensor = torch.tensor(class_index, dtype=Encoding.MOVE_DTYPE)
    #     return move_tensor

    # @staticmethod
    # def move_from_tensor(move_tensor: torch.Tensor, classindex_to_uci: dict[int, str]) -> str:
        """
        Converts a scalar move tensor representing a class index into its corresponding UCI move string.
        Args:
            move_tensor (torch.Tensor): A scalar tensor containing the class index of the move.
            classindex_to_uci (dict[int, str]): A mapping from class indices to UCI move strings.
        Returns:
            str: The UCI string representation of the move.
        Raises:
            AssertionError: If the input tensor shape is not scalar or if the encoding input shape is not as expected.
        """

        assert Encoding.INPUT_SHAPE[0] == 16, "not updated to new encoding"

        assert move_tensor.shape == (), f"move_tensor shape must be (), got {move_tensor.shape}"
        move_uci = classindex_to_uci[int(move_tensor.item())]
        return move_uci


if __name__ == "__main__":
    ###############################################################################
    #   Example usage
    #
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("h7h6"))

    print(f"Current board:\n{board}")

    plane_index = Encoding.PLANE.BLACK_ROOK
    board_tensor = Encoding.board_to_tensor(board)
    # board_tensor[plane_index, 7, 1] = 99

    print(f"Board tensor shape: {board_tensor.shape}")
    # print(f"One plane of the board:\n{board_tensor[plane_index, :, :]}")

    # rotated_tensor = torch.flip(input=board_tensor, dims=[1],)
    rotated_tensor = Encoding.board_tensor_flip(board_tensor, in_place=True)

    # rotated_tensor = EncodingUtils.flip_board_tensor(board_tensor)
    # print(f"Rotated plane of the board:\n{rotated_tensor[plane_index, :, :]}")

    board = Encoding.board_from_tensor(rotated_tensor)
    print(f"Reconstructed board:\n{board}")
