import chess

from src.libs.encoding import Encoding


if __name__ == "__main__":
    # Example usage
    board = chess.Board()
    board.push_uci("e2e4")
    print(f"Initial Board: turn {board.turn}")
    print(board)

    # Encode the board
    board_tensor = Encoding.board_to_tensor(board)
    # print("Encoded Board Tensor:")
    # print(board_tensor)

    print('ff')

    # Decode the tensor back to a board
    decoded_board = Encoding.board_from_tensor(board_tensor)
    print("Decoded Board:")
    print(decoded_board)

    # Verify that the original and decoded boards are the same
    # assert board.board_fen() == decoded_board.board_fen(), "The decoded board does not match the original board!"