import chess
import colorama




def board_to_string(board: chess.Board, flip_board: bool = False, borders: bool = True, bg_colors: bool = True) -> str:
    board_width = 8
    board_unicode = board.unicode()



    # board_str looks like this:
    # ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
    # ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
    # ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
    # ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
    # ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
    # ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
    # ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
    # ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖

    
    # Split the board into lines
    board_lines = board_unicode.split("\n")


    # Make each square 3 characters wide instead of 2
    # "♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜" -> " ♜  ♞  ♝  ♛  ♚  ♝  ♞  ♜ "
    board_lines = [line.replace(" ", "  ") for line in board_lines]
    board_lines = [f" {line} " for line in board_lines]
    
    # Replace "⭘" dots with empty squares
    board_lines = [line.replace("⭘", " ") for line in board_lines]

    if flip_board:
        board_lines = board_lines[::-1]

    # Add colors
    if bg_colors == True:
        white_bg =  colorama.Fore.BLACK + colorama.Back.YELLOW
        black_bg =  colorama.Fore.BLACK + colorama.Back.GREEN
        # white_bg = colorama.Fore.WHITE
        # black_bg = colorama.Fore.WHITE
        reset_color = colorama.Style.RESET_ALL
        for row in range(board_width):
            line = ""
            for col in range(board_width):
                char = board_lines[row][col * 3 + 1 : col * 3 + 2]
                char = " " + char + " "
                if (row + col) % 2 == (0 if not flip_board else 1):
                    line += f"{white_bg}{char}{reset_color}"
                else:
                    line += f"{black_bg}{char}{reset_color}"
            board_lines[row] = line

    # Honor borders option
    if borders == True:
        # Add side borders
        board_lines = [f"|{line}|" for line in board_lines]
        # Add the row numbers on the left
        row_numbers = list(range(8, 0, -1)) if not flip_board else list(range(1, 9))
        board_lines = [f"{row_numbers[i]}{line}" for i, line in enumerate(board_lines)]
        # Add top and bottom borders
        border_line = " " + "+" + "-" * (board_width * 3) + "+"
        board_lines = [border_line] + board_lines + [border_line]
        # Add the column letters at the bottom
        column_letters = "   A  B  C  D  E  F  G  H  " if not flip_board else "   H  G  F  E  D  C  B  A  "
        board_lines.append(column_letters)

    return "\n".join(board_lines)

board = chess.Board()

print(board)
board.push_uci("e2e4")
print(board.unicode())

print(board_to_string(board, flip_board=True))
