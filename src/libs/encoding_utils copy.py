import numpy as np
import torch
import chess
import chess.pgn as pgn
from tqdm import tqdm

from src.libs.chess_extra import ChessExtra


class EncodingUtils:

    INPUT_SHAPE = (16, 8, 8)  # (channels, height, width)
    BOARD_DTYPE = np.dtype(np.float32)
    MOVE_DTYPE = np.dtype(np.long)  # class index as long

    @staticmethod
    def __board_to_numpy_new(board: chess.Board) -> np.ndarray:
        # AlphaZero style board encoding
        # https://github.com/iamlucaswolf/gym-chess/blob/6a5eb43650c400a556505ec035cc3a3c5792f8b2/gym_chess/alphazero/board_encoding.py#L11C1-L33C1

        assert EncodingUtils.INPUT_SHAPE == (21, 8, 8), f"INPUT_SHAPE must be (21, 8, 8), got {EncodingUtils.INPUT_SHAPE}"

        unique_piece_count = 6  # P, N, B, R, Q, K
        color_count = 2  # white, black
        board_plane_count = unique_piece_count * color_count

        repetition_plane_count = 2  # planes for 2 and 3 fold repetition

        # 7 additional planes for metadata
        # - 1 plane for active player color
        # - 1 plane for total move count
        # - 2 planes for active player castling rights
        # - 2 planes for opponent player castling rights
        # - 1 plane for no-progress counter
        meta_plane_count = 7

        plane_count = board_plane_count + repetition_plane_count + meta_plane_count


        ###############################################################################
        #   Piece planes
        #
        piece_planes = np.zeros((board_plane_count, 8, 8))

        # Populate first 12 8x8 boards (where pieces are)
        for square, piece in board.piece_map().items():
            row, col = divmod(square, 8)
            piece_color = 0 if piece.color else 6
            piece_type = piece.piece_type - 1
            piece_planes[piece_color + piece_type, row, col] = 1

        ###############################################################################
        #   Repetition planes
        #
        # Repetition counters
        repetition_planes = np.zeros((repetition_plane_count, 8, 8))
        repetition_planes[0, :, :] = board.is_repetition(2)
        repetition_planes[1, :, :] = board.is_repetition(3)

        ###############################################################################
        #   Metadata planes
        #
        meta_planes = np.zeros(shape=(meta_plane_count, 8, 8))

        # Active player color
        meta_planes[0, :, :] = int(board.turn)  # 1 for white, 0 for black

        # Total move count
        meta_planes[1, :, :] = board.fullmove_number

        # # Active player castling rights
        active_color = board.turn
        meta_planes[2, :, :] = board.has_kingside_castling_rights(active_color)
        meta_planes[3, :, :] = board.has_queenside_castling_rights(active_color)

        # Opponent player castling rights
        opponent_color = not board.turn
        meta_planes[4, :, :] = board.has_kingside_castling_rights(opponent_color)
        meta_planes[5, :, :] = board.has_queenside_castling_rights(opponent_color)

        # # No-progress counter
        meta_planes[6, :, :] = board.halfmove_clock

        ###############################################################################
        #   Concatenate all planes
        #
        board_numpy = np.concatenate([piece_planes, repetition_planes, meta_planes], axis=0)

        # Return a numpy array of shape (21, 8, 8)
        return board_numpy

    @staticmethod
    def __numpy_dtype_to_torch(np_dtype: np.dtype) -> torch.dtype:
        if np_dtype == np.float32:
            return torch.float32
        elif np_dtype == np.float64:
            return torch.float64
        elif np_dtype == np.uint32:
            return torch.uint32
        elif np_dtype == np.uint64:
            return torch.uint64
        elif np_dtype == np.uint16:
            return torch.uint16
        elif np_dtype == np.uint8:
            return torch.uint8
        elif np_dtype == np.bool_:
            return torch.bool
        else:
            raise ValueError(f"No torch equivalent for numpy dtype {np_dtype}")

    @staticmethod
    def __board_to_numpy(board: chess.Board) -> np.ndarray:
        # Create a numpy array of shape (16, 8, 8)
        # 8x8 is a size of the chess board.
        # 12 = number of unique pieces.
        # 13th board for pieces we can move (FROM WHERE we can move)
        # 14th board for legal moves (WHERE we can move)
        # 15th board for number of our pieces attacking each square
        # 16th board for number of opponent pieces attacking each square

        assert EncodingUtils.INPUT_SHAPE == (16, 8, 8), f"INPUT_SHAPE must be (16, 8, 8), got {EncodingUtils.INPUT_SHAPE}"

        board_np = np.zeros((16, 8, 8))
        piece_map = board.piece_map()

        # Populate first 12 8x8 boards (where pieces are)
        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            piece_color = 0 if piece.color else 6
            piece_type = piece.piece_type - 1
            board_np[piece_color + piece_type, row, col] = 1

        # Populate the 13th board - squares FROM WHICH we can move (i.e. pieces we can move)
        legal_moves = board.legal_moves
        for move in legal_moves:
            from_square = move.from_square
            row_from, col_from = divmod(from_square, 8)
            board_np[12, row_from, col_from] = 1

        # Populate the 14th board - squares TO WHICH we can move (i.e. legal move targets)
        for move in legal_moves:
            to_square = move.to_square
            row_to, col_to = divmod(to_square, 8)
            board_np[13, row_to, col_to] = 1

        # Populate the 15th board - number of our pieces attacking each square
        my_color = board.turn
        opponent_color = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
        my_board_attacked_count = ChessExtra.board_attacked_count_compute(board, opponent_color)
        board_np[14] = np.array(my_board_attacked_count)

        # Populate the 16th board - number of opponent pieces attacking each square
        opponent_board_attacked_count = ChessExtra.board_attacked_count_compute(board, my_color)
        board_np[15] = np.array(opponent_board_attacked_count)

        # Return a numpy array of shape (16, 8, 8)
        return board_np

    @staticmethod
    def board_to_tensor(board: chess.Board) -> torch.Tensor:
        board_np = EncodingUtils.__board_to_numpy(board)

        board_tensor_dtype = EncodingUtils.__numpy_dtype_to_torch(EncodingUtils.BOARD_DTYPE)
        board_tensor = torch.tensor(board_np, dtype=board_tensor_dtype).unsqueeze(0)
        return board_tensor

    @staticmethod
    def move_to_tensor(move_uci: str, uci_to_classindex: dict[str, int]) -> torch.Tensor:
        """
        convert a move in UCI string format to a tensor (scalar tensor with class index) using the provided mapping
        """

        class_index = uci_to_classindex[move_uci]
        moves_tensor_dtype = EncodingUtils.__numpy_dtype_to_torch(EncodingUtils.MOVE_DTYPE)
        move_tensor = torch.tensor(class_index, dtype=moves_tensor_dtype)
        return move_tensor

    @staticmethod
    def board_from_tensor(board_tensor: torch.Tensor) -> chess.Board:
        """
        convert a board tensor of shape (16, 8, 8) to a chess.Board object
        """

        assert False, "not updated to new encoding"

        assert board_tensor.shape == (16, 8, 8), f"board_tensor shape must be (16, 8, 8), got {board_tensor.shape}"
        board_recontruct = chess.Board()
        board_recontruct.clear_board()

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
                    board_recontruct.set_piece_at(square, piece)
        return board_recontruct

    @staticmethod
    def move_from_tensor(move_tensor: torch.Tensor, classindex_to_uci: dict[int, str]) -> str:
        """
        convert a move tensor (scalar tensor with class index) to UCI string using the provided mapping
        """

        assert False, "not updated to new encoding"

        assert move_tensor.shape == (), f"move_tensor shape must be (), got {move_tensor.shape}"
        move_uci = classindex_to_uci[int(move_tensor.item())]
        return move_uci

    @staticmethod
    def games_to_tensor(games: list[pgn.Game]) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:


        ###############################################################################
        #   Count total positions
        #
        position_count = 0
        for game in games:
            game_move_count = len(list(game.mainline_moves()))
            position_count += game_move_count

        boards_native =[]
        moves_uci = []
        for game in tqdm(games, ncols=80, desc="Encoding", unit="games"):
            board = game.board()
            for move in game.mainline_moves():
                # encode the current board position
                board_numpy = EncodingUtils.__board_to_numpy(board)
                boards_native.append(board_numpy)

                # append the best move in UCI format
                moves_uci.append(move.uci())

                # Play this move on the board to get to the next position
                board.push(move)

        # boards_numpy = np.array(boards_native, dtype=EncodingUtils.BOARD_DTYPE)
        board_tensor_dtype = EncodingUtils.__numpy_dtype_to_torch(EncodingUtils.BOARD_DTYPE)
        boards_tensor = torch.tensor(boards_native, dtype=board_tensor_dtype)

        # Create a mapping from UCI move strings to class indices
        uci_to_classindex = {move: idx for idx, move in enumerate(set(moves_uci))}

        # Encode best moves as class indices
        moves_tensor_dtype = EncodingUtils.__numpy_dtype_to_torch(EncodingUtils.MOVE_DTYPE)
        moves_tensor = torch.tensor([uci_to_classindex[move] for move in moves_uci], dtype=moves_tensor_dtype)

        # return the boards, best moves and the mapping
        return boards_tensor, moves_tensor, uci_to_classindex

if __name__ == "__main__":
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    # board.push(chess.Move.from_uci("e7e5"))

    print(f"Current board:\n{board}")
    board_numpy = EncodingUtils.__board_to_numpy(board)
    print(f"Board tensor shape: {board_numpy.shape}")
