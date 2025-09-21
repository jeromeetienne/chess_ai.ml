# pip imports
import chess.pgn

class ChessExtra:
    """
    Extra python-chess utilities -- https://python-chess.readthedocs.io/en/latest/
    """

    @staticmethod    
    def print_game(game: chess.pgn.Game) -> None:
        print(game.board().unicode())
        for move in game.mainline_moves():
            print(f"move: {move.uci()}")

    
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
