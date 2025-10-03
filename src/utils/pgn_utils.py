# stdlib imports
import datetime
import os
from typing import Iterator

# pip imports
import chess
import chess.pgn 

__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "../../data")
pgn_folder_path = os.path.join(data_folder_path, "pgn_splits")

class PGNUtils:
    @staticmethod
    def get_pgn_paths() -> list[str]:
        """
        Return a list of all PGN file paths in the specified folder.
        The list is sorted alphabetically.

        Arguments:
            folder_path (str): Path to the folder containing PGN files.
        """
        
        basenames = [basename for basename in os.listdir(pgn_folder_path) if basename.endswith(".pgn")]
        pgn_paths = [os.path.join(pgn_folder_path, basename) for basename in basenames]
        pgn_paths.sort()

        return pgn_paths

    @staticmethod
    def parse_pgn_file_iter(file_path: str) -> Iterator[chess.pgn.Game]:
        """
        Parse a PGN file and yield chess.pgn.Game objects one by one.

        Arguments:
            file_path (str): Path to the PGN file.
        """
        with open(file_path, 'r') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                yield game


    @staticmethod
    def parse_pgn_file(file_path: str) -> list[chess.pgn.Game]:
        """
        Parse a PGN file and return a list of chess.pgn.Game objects.

        Arguments:
            file_path (str): Path to the PGN file.
        """
        games: list[chess.pgn.Game] = []
        with open(file_path, 'r') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
        return games

    @staticmethod
    def board_to_pgn(board: chess.Board, white_player: str = "Unknown", black_player: str = "Unknown") -> str:
        """
        Convert a chess.Board to a PGN string with headers.

        Arguments:
            board (chess.Board): The chess board to convert.
            white_player (str): Name of the white player.
            black_player (str): Name of the black player.
        """
        game = chess.pgn.Game.from_board(board)
        game.headers["Event"] = "AI vs AI"
        game.headers["White"] = white_player
        game.headers["Black"] = black_player
        game.headers["Result"] = board.result()

        game.headers["Site"] = "My Computer"
        game.headers["Round"] = "1"
        # set the date to today in YYYY.MM.DD format
        today = datetime.date.today()
        game.headers["Date"] = today.strftime("%Y.%m.%d")

        pgn_game = game.__str__()
        return pgn_game
