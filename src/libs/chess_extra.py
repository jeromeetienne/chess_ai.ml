# pip imports
import chess.pgn
import chess.polyglot
import colorama

class ChessExtra:
    """
    Extra python-chess utilities -- https://python-chess.readthedocs.io/en/latest/
    """

    @staticmethod
    def is_in_opening_book(board: chess.Board, polyglot_reader: chess.polyglot.MemoryMappedReader) -> bool:
        """
        Check if the given board state is out of the opening book.
        Args:
            board (chess.Board): The current state of the chess board.
        Returns:
            bool: True if the board state is in the opening book, False otherwise.
        """
        try:
            polyglot_reader.weighted_choice(board)
            return True  # Found a move in the opening book
        except IndexError:
            return False  # No move found in the opening book

    @staticmethod
    def game_slice(src_game: chess.pgn.Game, move_index_start: int, move_index_end: int) -> chess.pgn.Game:
        """
        Extract a slice of a game from move_index_start to move_index_end (exclusive).
        The move indices are 0-based.
        Returns a new chess.pgn.Game object containing only the specified moves.
        
        Arguments:
        - src_game (chess.pgn.Game): The source chess.pgn.Game object to slice.
        - move_index_start (int): The starting move index (inclusive).
        - move_index_end (int): The ending move index (exclusive).
        """

        src_mainline_moves = list(src_game.mainline_moves())
        # print(f"Total moves in first game: {len(src_mainline_moves)}")
        # print(f"first 5 moves: {[move.uci() for move in src_mainline_moves[:5]]}")

        # sanity check
        assert len(src_mainline_moves) >= move_index_end, f"Game has only {len(src_mainline_moves)} moves, cannot extract moves {move_index_start} to {move_index_end}"
        assert move_index_start < move_index_end, f"move_index_start ({move_index_start}) must be less than move_index_end ({move_index_end})"

        # print(f"Total moves in first game: {len(src_mainline_moves)}")

        tmp_board = chess.Board()
        for move in src_mainline_moves[:move_index_start]:
            tmp_board.push_uci(move.uci())

        # create a new board
        tmp_board_fen = tmp_board.fen()
        dst_board = chess.Board(tmp_board_fen)

        for move in src_mainline_moves[move_index_start:move_index_end]:
            # print(f"applying move {move.uci()}")
            dst_board.push_uci(move.uci())

        dst_game = chess.pgn.Game.from_board(dst_board)
        return dst_game

    @staticmethod
    def board_to_string(board: chess.Board, flip_board: bool = False, borders: bool = True, colors: bool = True) -> str:
        """
        Convert a chess.Board to a string representation with optional flipping, borders, and background colors.

        Args:
            board (chess.Board): The chess board to convert.
            flip_board (bool): Whether to flip the board vertically.
            borders (bool): Whether to include borders around the board.
            colors (bool): Whether to include background colors for squares.

        Returns:
            str: The string representation of the board.
        """

        board_width = 8
        board_unicode = board.unicode()

        # board_unicode looks like this:
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

        # honor flip_board option
        if flip_board:
            # Flip the board vertically
            board_lines = board_lines[::-1]
            # Flip the pieces horizontally
            board_lines = [line[::-1] for line in board_lines]

        # honor colors option
        if colors == True:
            white_bg =  colorama.Fore.BLACK + colorama.Back.YELLOW
            black_bg =  colorama.Fore.BLACK + colorama.Back.GREEN
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
            board_lines = [f"│{line}│" for line in board_lines]
            # Add the row numbers on the left
            row_numbers = list(range(8, 0, -1)) if not flip_board else list(range(1, 9))
            board_lines = [f"{row_numbers[i]}{line}" for i, line in enumerate(board_lines)]
            # Add top and bottom borders
            border_top_line = " " + "┌" + "─" * (board_width * 3) + "┐"
            border_bottom_line = " " + "└" + "─" * (board_width * 3) + "┘"
            board_lines = [border_top_line] + board_lines + [border_bottom_line]
            # Add the column letters at the bottom
            column_letters = "   A  B  C  D  E  F  G  H  " if not flip_board else "   H  G  F  E  D  C  B  A  "
            board_lines.append(column_letters)

        # add a line "Turn: White" or "Turn: Black" on top of the board
        turn_str = "White" if board.turn == chess.WHITE else "Black"
        board_lines = [f" Turn: {turn_str}"] + board_lines

        return "\n".join(board_lines)

    @staticmethod
    def piece_unique_moves(piece_symbol: str, turn: chess.Color) -> set[str]:
        """
        Count all possible moves for a given piece type on an empty chess board.
        """
        # Create empty board
        board = chess.Board()

        move_count = 0
        unique_moves: set[str] = set()

        # Special case for kings: Handle castling moves
        if piece_symbol == 'K': # white king
            unique_moves.add('e1g1') # white king side
            unique_moves.add('e1c1') # white queen side
        elif piece_symbol == 'k': # black king
            unique_moves.add('e8g8') # black king side
            unique_moves.add('e8c8') # black queen side

        # enumerate all the squares
        for square in chess.SQUARES:
            # clear the board
            board.clear()
            # put a piece on the square
            board.set_piece_at(square, chess.Piece.from_symbol(piece_symbol))
            board.turn = turn

            ###############################################################################
            #   Special cases
            #

            # Special case for pawns: they cannot be on the first or last rank
            if(piece_symbol == 'P'): # white pawn
                # pawns cannot be on the first rank
                if chess.square_rank(square) == 0:
                    continue
            elif(piece_symbol == 'p'): # black pawn
                # pawns cannot be on the last rank
                if chess.square_rank(square) == 7:
                    continue

            # Special case for pawns: Handle pawns special moves to capture diagonally
            # - add 2 pawns from the other color as bait to attack diagonally
            if piece_symbol == 'P' and chess.square_rank(square) != 7:
                # on file A, cannot add a pawn on the left diagonal
                if chess.square_file(square) != 0:
                    board.set_piece_at(square+7, chess.Piece.from_symbol('p'))
                # on file H, cannot add a pawn on the right diagonal
                if chess.square_file(square) != 7:
                    board.set_piece_at(square+9, chess.Piece.from_symbol('p'))
            elif piece_symbol == 'p' and chess.square_rank(square) != 0:
                # on file A, cannot add a pawn on the left diagonal
                if chess.square_file(square) != 0:
                    board.set_piece_at(square-9, chess.Piece.from_symbol('P'))
                # on file H, cannot add a pawn on the right diagonal
                if chess.square_file(square) != 7:
                    board.set_piece_at(square-7, chess.Piece.from_symbol('P'))

            ###############################################################################
            #   Process the board
            #

            # Get all possible moves for the rook
            possible_moves = list(board.legal_moves)
            move_count += len(possible_moves)
            for move in possible_moves:
                unique_moves.add(move.uci())

        return unique_moves

    @staticmethod
    def all_unique_moves() -> set[str]:
        """
        Count all possible moves for all piece types on an empty chess board.
        
        FIXME everybody say it is 1972. but i count only 1968. Where are the missing 4 !
        """

        total_unique_moves = set()

        # enumerate all piece types
        piece_types = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']

        for piece_type in piece_types:
            # print(f"Counting moves for piece type: {piece_type}")
            turn = chess.WHITE if piece_type.isupper() else chess.BLACK
            # get unique moves for this piece type
            piece_unique_moves = ChessExtra.piece_unique_moves(piece_type, turn)
            # update the total unique moves set
            total_unique_moves.update(piece_unique_moves)

        return total_unique_moves

    @staticmethod
    def board_attacked_count_compute(board: chess.Board, color: chess.Color) -> list[list[int]]:
        """
        For each square on the board, count how many pieces of the opposite color are attacking it.
        Return a 2D array of shape (8, 8) with the counts.
        """
        # initialize the 8x8 array with zeros
        board_square_count = [[0 for _ in range(8)] for _ in range(8)]
        for square in chess.SQUARES:
            attackers_squareset = board.attackers(color, square)
            attackers_list = list(attackers_squareset)
            if len(attackers_list) == 0:
                continue
            # print(f"Square {chess.square_name(square)} is attacked by {len(attackers_list)} pieces: {[chess.square_name(sq) for sq in attackers_list]}")
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            board_square_count[rank][file] += len(attackers_list)

        return board_square_count

    @staticmethod
    def board_square_count_to_string(board_square_count: list[list[int]]) -> str:
        """
        Print the attacked_squares 2D array in a readable format.
        """
        output = []
        output.append("Attacked squares (number of attackers):")
        output.append("  +-----------------------+")
        for rank in range(7, -1, -1):
            line = f"{rank + 1}|"
            for file in range(8):
                line += f" {board_square_count[rank][file]} "
            line += f"|"
            output.append(line)
        output.append("  +-----------------------+")
        output.append("   A  B  C  D  E  F  G  H")
        return "\n".join(output)
