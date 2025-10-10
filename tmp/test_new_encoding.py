import chess

from src.encoding.board_encoding import BoardEncoding


if __name__ == "__main__":
    # Example usage
    board = chess.Board()
    board.push_uci("e2e4")
    print(f"Initial Board: turn {board.turn}")
    print(board)

    # Encode the board
    board_tensor = BoardEncoding.board_to_tensor(board)
    # print("Encoded Board Tensor:")
    # print(board_tensor)

    print("ff")

    # Decode the tensor back to a board
    decoded_board = BoardEncoding.board_from_tensor(board_tensor)
    print("Decoded Board:")
    print(decoded_board)

    # Verify that the original and decoded boards are the same
    # assert board.board_fen() == decoded_board.board_fen(), "The decoded board does not match the original board!"
