import numpy as np
import torch
import chess
import chess.pgn as pgn

class EncodingUtils:
    @staticmethod
    def board_to_matrix_np(board: chess.Board) -> np.ndarray:
        # 8x8 is a size of the chess board.
        # 12 = number of unique pieces.
        # 13th board for pieces we can move (FROM WHERE we can move)
        # 14th board for legal moves (WHERE we can move)
        matrix = np.zeros((14, 8, 8))
        piece_map = board.piece_map()

        # Populate first 12 8x8 boards (where pieces are)
        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            piece_type = piece.piece_type - 1
            piece_color = 0 if piece.color else 6
            matrix[piece_type + piece_color, row, col] = 1

        # Populate the 13th board - squares FROM WHICH we can move (i.e. pieces we can move)
        legal_moves = board.legal_moves
        for move in legal_moves:
            from_square = move.from_square
            row_from, col_from = divmod(from_square, 8)
            matrix[12, row_from, col_from] = 1

        # Populate the 14th board - squares TO WHICH we can move (i.e. legal move targets)
        for move in legal_moves:
            to_square = move.to_square
            row_to, col_to = divmod(to_square, 8)
            matrix[13, row_to, col_to] = 1

        return matrix

    @staticmethod
    def board_to_tensor(board: chess.Board) -> torch.Tensor:
        matrix = EncodingUtils.board_to_matrix_np(board)
        board_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        return board_tensor

    @staticmethod
    def create_input_for_nn_np(games: list[pgn.Game]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        board_array = []
        best_move_array = []
        for game in games:
            board = game.board()
            for move in game.mainline_moves():
                board_matrix = EncodingUtils.board_to_matrix_np(board)
                board_array.append(board_matrix)
                best_move_array.append(move.uci())
                board.push(move)

        board_nparray =  np.array(board_array, dtype=np.float32)

        # Create a mapping from UCI move strings to class indices
        uci_to_classindex = {move: idx for idx, move in enumerate(set(best_move_array))}

        # Encode best moves as class indices
        best_move_nparray = np.array([uci_to_classindex[move] for move in best_move_array], dtype=np.float32)


        return board_nparray, best_move_nparray, uci_to_classindex

    # @staticmethod   
    # def encode_moves_np(moves: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    #     uci_to_classindex = {move: idx for idx, move in enumerate(set(moves))}
    #     return np.array([uci_to_classindex[move] for move in moves], dtype=np.float32), uci_to_classindex
    
