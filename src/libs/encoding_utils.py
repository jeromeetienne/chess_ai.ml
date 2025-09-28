import numpy as np
import torch
import chess
import chess.pgn as pgn
from tqdm import tqdm

from .chess_extra import ChessExtra


class EncodingUtils:
    @staticmethod
    def board_to_matrix_np(board: chess.Board) -> np.ndarray:
        # Create a numpy array of shape (16, 8, 8)
        # 8x8 is a size of the chess board.
        # 12 = number of unique pieces.
        # 13th board for pieces we can move (FROM WHERE we can move)
        # 14th board for legal moves (WHERE we can move)
        # 15th board for number of our pieces attacking each square
        # 16th board for number of opponent pieces attacking each square
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
        matrix = EncodingUtils.board_to_matrix_np(board)
        board_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        return board_tensor

    @staticmethod
    def board_from_tensor(board_tensor: torch.Tensor) -> chess.Board:
        """
        convert a board tensor of shape (16, 8, 8) to a chess.Board object
        """
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
        assert move_tensor.shape == (), f"move_tensor shape must be (), got {move_tensor.shape}"
        move_uci = classindex_to_uci[int(move_tensor.item())]
        return move_uci



    @staticmethod
    def create_input_for_nn_np(games: list[pgn.Game]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        board_array = []
        best_move_array_uci = []
        for game in tqdm(games, ncols=80, desc="Encoding", unit="games"):
            board = game.board()
            for move in game.mainline_moves():
                # encode the current board position
                board_matrix = EncodingUtils.board_to_matrix_np(board)
                board_array.append(board_matrix)
                # append the best move in UCI format
                best_move_array_uci.append(move.uci())

                # Play this move on the board to get to the next position
                board.push(move)

        board_nparray = np.array(board_array, dtype=np.float32)

        # Create a mapping from UCI move strings to class indices
        uci_to_classindex = {move: idx for idx, move in enumerate(set(best_move_array_uci))}

        # Encode best moves as class indices
        best_move_nparray = np.array([uci_to_classindex[move] for move in best_move_array_uci], dtype=np.float32)

        # return the boards, best moves and the mapping
        return board_nparray, best_move_nparray, uci_to_classindex

    # @staticmethod
    # def encode_moves_np(moves: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    #     uci_to_classindex = {move: idx for idx, move in enumerate(set(moves))}
    #     return np.array([uci_to_classindex[move] for move in moves], dtype=np.float32), uci_to_classindex
