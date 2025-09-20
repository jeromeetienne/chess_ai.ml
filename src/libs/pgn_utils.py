# stdlib imports
import datetime

# pip imports
import chess
import chess.pgn 

class PGNUtils:
    @staticmethod
    def load_games_from_pgn(file_path: str) -> list[chess.pgn.Game]:
        games: list[chess.pgn.Game] = []
        with open(file_path, 'r') as pgn_file:
            while True:
                game: chess.pgn.Game|None = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)

        return games

    @staticmethod
    def board_to_pgn(board: chess.Board, white_player: str = "Unknown", black_player: str = "Unknown") -> str:
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
