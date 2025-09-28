# pip imports
import numpy as np
import torch
import chess
import chess.pgn as pgn
from tqdm import tqdm

# local imports
from src.libs.chess_extra import ChessExtra

class Encoding:

    INPUT_SHAPE = (21, 8, 8)  # (channels, height, width)
    # INPUT_SHAPE = (16, 8, 8)  # (channels, height, width)
    BOARD_DTYPE =torch.float32
    MOVE_DTYPE =torch.long  # class index as long
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
        ACTIVE_KINGSIDE_CASTLING_RIGHTS = 16
        ACTIVE_QUEENSIDE_CASTLING_RIGHTS = 17
        OPPONENT_KINGSIDE_CASTLING_RIGHTS = 18
        OPPONENT_QUEENSIDE_CASTLING_RIGHTS = 19
        NO_PROGRESS_COUNTER = 20

    @staticmethod
    def board_to_tensor(board: chess.Board) -> torch.Tensor:
        # AlphaZero style board encoding
        # https://github.com/iamlucaswolf/gym-chess/blob/6a5eb43650c400a556505ec035cc3a3c5792f8b2/gym_chess/alphazero/board_encoding.py#L11C1-L33C1

        assert Encoding.INPUT_SHAPE == (21, 8, 8), f"INPUT_SHAPE must be (21, 8, 8), got {Encoding.INPUT_SHAPE}"

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

        # board_tensor = torch.zeros(plane_count, 8, 8, dtype=Encoding.BOARD_DTYPE)

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

        # Convert the numpy array to a torch tensor
        # board_tensor_dtype = EncodingUtils.__numpy_dtype_to_torch(EncodingUtils.BOARD_DTYPE)
        board_tensor = torch.tensor(board_numpy, dtype=Encoding.BOARD_DTYPE)

        return board_tensor

    @staticmethod
    def board_from_tensor(board_tensor: torch.Tensor) -> chess.Board:
        """
        convert a board tensor of shape (16, 8, 8) to a chess.Board object
        """

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

        # TMP: remove this when not needed
        if Encoding.INPUT_SHAPE[0] == 16:
            return board_recontruct


        # set turn
        board_recontruct.turn = bool(board_tensor[12, 0, 0].item())
        # set fullmove number
        board_recontruct.fullmove_number = int(board_tensor[13, 0, 0].item())
        # set castling rights
        if board_tensor[14, 0, 0].item():
            if board_recontruct.turn == chess.WHITE:
                board_recontruct.castling_rights |= chess.BB_H1
            else:
                board_recontruct.castling_rights |= chess.BB_A8
        if board_tensor[15, 0, 0].item():
            if board_recontruct.turn == chess.WHITE:
                board_recontruct.castling_rights |= chess.BB_A1
            else:
                board_recontruct.castling_rights |= chess.BB_H8
        # set halfmove clock
        board_recontruct.halfmove_clock = int(board_tensor[19, 0, 0].item())

        return board_recontruct

    @staticmethod
    def board_tensor_flip(board_tensor: torch.Tensor, in_place: bool = False) -> torch.Tensor:
        """
        Flip the board tensor to switch perspective between white and black.
        This involves rotating the piece planes and swapping the piece colors.
        """

        assert board_tensor.shape == Encoding.INPUT_SHAPE, f"board_tensor shape must be {Encoding.INPUT_SHAPE}, got {board_tensor.shape}"

        # Rotate all planes by 180 degrees
        rotated_tensor = torch.flip(input=board_tensor, dims=[1,2])

        if in_place == False:
            return rotated_tensor

        board_tensor.copy_(rotated_tensor)
        return board_tensor

    @staticmethod
    def move_to_tensor(move_uci: str, uci_to_classindex: dict[str, int]) -> torch.Tensor:
        """
        convert a move in UCI string format to a tensor (scalar tensor with class index) using the provided mapping
        """
        # convert move in UCI format to class index
        class_index = uci_to_classindex[move_uci]
        # convert class index to tensor
        # moves_tensor_dtype = EncodingUtils.__numpy_dtype_to_torch(EncodingUtils.MOVE_DTYPE)
        move_tensor = torch.tensor(class_index, dtype=Encoding.MOVE_DTYPE)
        return move_tensor

    @staticmethod
    def move_from_tensor(move_tensor: torch.Tensor, classindex_to_uci: dict[int, str]) -> str:
        """
        convert a move tensor (scalar tensor with class index) to UCI string using the provided mapping
        """

        assert Encoding.INPUT_SHAPE[0] == 16, "not updated to new encoding"

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

        boards_tensor = torch.zeros((position_count, *Encoding.INPUT_SHAPE), dtype=Encoding.BOARD_DTYPE)
        moves_uci = []
        board_index = 0
        for game in tqdm(games, ncols=80, desc="Encoding", unit="games"):
            board = game.board()
            for move in game.mainline_moves():
                # encode the current board position
                board_tensor = Encoding.board_to_tensor(board)
                boards_tensor[board_index] = board_tensor

                # append the best move in UCI format
                moves_uci.append(move.uci())

                # Play this move on the board to get to the next position
                board.push(move)
                # Update the board index
                board_index += 1

        # boards_numpy = np.array(boards_native, dtype=EncodingUtils.BOARD_DTYPE)
        # board_tensor_dtype = EncodingUtils.__numpy_dtype_to_torch(EncodingUtils.BOARD_DTYPE)
        # boards_tensor = torch.tensor(boards_tensor, dtype=board_tensor_dtype)

        # Create a mapping from UCI move strings to class indices
        uci_to_classindex = {move: idx for idx, move in enumerate(set(moves_uci))}

        # Encode best moves as class indices
        # moves_tensor_dtype = EncodingUtils.__numpy_dtype_to_torch(EncodingUtils.MOVE_DTYPE)
        moves_array = [uci_to_classindex[move] for move in moves_uci]
        moves_tensor = torch.tensor(moves_array, dtype=Encoding.MOVE_DTYPE)

        # return the boards, best moves and the mapping
        return boards_tensor, moves_tensor, uci_to_classindex

if __name__ == "__main__":
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("h7h6"))

    print(f"Current board:\n{board}")
    # board_numpy = EncodingUtils._board_to_numpy(board)
    # print(f"Board tensor shape: {board_numpy.shape}")

    # # print first plane of the board
    # # print(f"First plane of the board:\n{board_numpy[EncodingUtils.PLANES.WHITE_PAWN]}")

    # plane_index = EncodingUtils.PLANE.WHITE_PAWN

    # print(f"One plane of the board:\n{board_numpy[plane_index]}")

    # 
    # to_rotate = board_numpy[:, :, :]

    # rotated = np.rot90(board_numpy, k=2, axes=(1, 2))

    # print(f"Rotated plane of the board:\n{rotated[plane_index]}")


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