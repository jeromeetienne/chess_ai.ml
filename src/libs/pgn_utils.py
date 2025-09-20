import chess
import chess.pgn as pgn

class PGNUtils:
    def load_games_from_pgn(file_path: str) -> list[pgn.Game]:
        games: list[pgn.Game] = []
        with open(file_path, 'r') as pgn_file:
            while True:
                game: pgn.Game = pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
        return games

    def board_to_pgn(board: chess.Board) -> pgn.Game:
        game = pgn.Game.from_board(board)
        game.headers["Event"] = "AI vs AI"
        game.headers["White"] = "chess_bot.ml 1"
        game.headers["Black"] = "chess_bot.ml 2"
        game.headers["Result"] = board.result()

        game.headers["Site"] = "My Computer"
        game.headers["Round"] = "1"
        # set the date to today in YYYY.MM.DD format
        import datetime

        today = datetime.date.today()
        game.headers["Date"] = today.strftime("%Y.%m.%d")

        pgn_game = game.__str__()
        return pgn_game
