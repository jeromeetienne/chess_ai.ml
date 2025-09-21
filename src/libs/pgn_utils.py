# stdlib imports
import datetime
import os

# pip imports
import chess
import chess.pgn 

class PGNUtils:
    @staticmethod
    def all_pgn_file_paths(folder_path: str) -> list[str]:
        """
        Return a list of all PGN file paths in the specified folder.
        The list is sorted alphabetically.

        Arguments:
            folder_path (str): Path to the folder containing PGN files.
        """
        
        pgn_file_paths: list[str] = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pgn"):
                file_path = os.path.join(folder_path, file_name)
                pgn_file_paths.append(file_path)
        # sort the list
        pgn_file_paths.sort()

        return pgn_file_paths
    
    @staticmethod
    def load_games_from_pgn(file_path: str, max_games: int = 0) -> list[chess.pgn.Game]:
        """
        Load all games from a PGN file.

        Arguments:
            file_path (str): Path to the PGN file.
            max_games (int): Maximum number of games to load. If 0, load all games.
        """

        games: list[chess.pgn.Game] = []
        with open(file_path, 'r') as pgn_file:
            while True:
                if max_games != 0 and len(games) >= max_games:
                    break
                game: chess.pgn.Game|None = chess.pgn.read_game(pgn_file)
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
