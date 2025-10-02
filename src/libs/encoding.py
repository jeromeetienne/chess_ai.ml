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

        assert Encoding.INPUT_SHAPE == (21, 8, 8), f"INPUT_SHAPE must be (21, 8, 8), got {Encoding.INPUT_SHAPE}"

        active_color = board.turn
        opponent_color = chess.WHITE if active_color == chess.BLACK else chess.BLACK

        # create an empty tensor - Use numpy first, it is 3-4 faster than torch for this
        board_numpy = np.zeros(Encoding.INPUT_SHAPE, dtype=np.uint8)

        ###############################################################################
        #   Piece planes
        #

        # Populate first 12 8x8 boards (where pieces are)
        for square, piece in board.piece_map().items():
            row, col = divmod(square, 8)
            piece_color = 0 if piece.color == board.turn else 6
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

        # White player castling rights
        board_numpy[Encoding.PLANE.ACTIVE_KINGSIDE_CASTLING_RIGHTS, :, :] = board.has_kingside_castling_rights(board.turn)
        board_numpy[Encoding.PLANE.ACTIVE_QUEENSIDE_CASTLING_RIGHTS, :, :] = board.has_queenside_castling_rights(board.turn)

        # Black player castling rights
        board_numpy[Encoding.PLANE.OPPONENT_KINGSIDE_CASTLING_RIGHTS, :, :] = board.has_kingside_castling_rights(opponent_color)
        board_numpy[Encoding.PLANE.OPPONENT_QUEENSIDE_CASTLING_RIGHTS, :, :] = board.has_queenside_castling_rights(opponent_color)

        # Half-move clock
        board_numpy[Encoding.PLANE.HALFMOVE_CLOCK, :, :] = board.halfmove_clock

        # Full move number
        board_numpy[Encoding.PLANE.FULLMOVE_NUMBER, :, :] = board.fullmove_number

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

        active_color = bool(board_tensor[Encoding.PLANE.TURN, 0, 0].item())
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
                    # compute square
                    square = chess.square(file, rank)
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
        board.turn = bool(board_tensor[Encoding.PLANE.TURN, 0, 0].item())
        # set fullmove number
        board.fullmove_number = int(board_tensor[Encoding.PLANE.FULLMOVE_NUMBER, 0, 0].item())
        # set castling rights
        castling_fen = ""
        if active_color == chess.WHITE:
            if board_tensor[Encoding.PLANE.ACTIVE_KINGSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "K"
            if board_tensor[Encoding.PLANE.ACTIVE_QUEENSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "Q"
            if board_tensor[Encoding.PLANE.OPPONENT_KINGSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "k"
            if board_tensor[Encoding.PLANE.OPPONENT_QUEENSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "q"
        else:
            if board_tensor[Encoding.PLANE.OPPONENT_KINGSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "K"
            if board_tensor[Encoding.PLANE.OPPONENT_QUEENSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "Q"
            if board_tensor[Encoding.PLANE.ACTIVE_KINGSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "k"
            if board_tensor[Encoding.PLANE.ACTIVE_QUEENSIDE_CASTLING_RIGHTS, 0, 0].item():
                castling_fen += "q"
        if castling_fen == "":
            castling_fen = "-"
        board.set_castling_fen(castling_fen)
        # set halfmove clock
        board.halfmove_clock = int(board_tensor[Encoding.PLANE.HALFMOVE_CLOCK, 0, 0].item())

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

    @staticmethod
    def move_to_tensor(move_uci: str, color: chess.Color) -> torch.Tensor:
        uci2class = Uci2ClassUtils.get_uci2class(color)
        class_index = uci2class[move_uci]
        move_tensor = torch.tensor(class_index, dtype=Encoding.MOVE_DTYPE)
        return move_tensor

    @staticmethod
    def move_from_tensor(moves_tensor: torch.Tensor, color: chess.Color) -> str:
        """
        Converts a scalar move tensor representing a class index into its corresponding UCI move string.

        Arguments:
            moves_tensor (torch.Tensor): A scalar tensor containing the class index of the move.
            color (chess.Color): The color of the player making the move (chess.WHITE or chess.BLACK).
        Returns:
            str: The UCI string representation of the move.
        """
        class2uci = Uci2ClassUtils.get_class2uci(color)
        class_index = int(moves_tensor.item())
        move_uci = class2uci[class_index]
        return move_uci

if __name__ == "__main__":
    ###############################################################################
    #   Example usage
    #
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    move = chess.Move.from_uci("h7h6")
    print(f"Current board: {'white' if board.turn == chess.WHITE else 'black'} move {move.uci()}\n{board}")

    board_tensor = Encoding.board_to_tensor(board)
    move_tensor = Encoding.move_to_tensor(move.uci(), board.turn)


    reconstructed_board = Encoding.board_from_tensor(board_tensor)
    reconstructed_move_uci = Encoding.move_from_tensor(move_tensor, board.turn)
    reconstructed_move = chess.Move.from_uci(reconstructed_move_uci)
    print(f"Reconstructed board: {'white' if reconstructed_board.turn == chess.WHITE else 'black'} move {reconstructed_move.uci()}\n{reconstructed_board}")