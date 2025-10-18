#!/usr/bin/env python3

# stdlib imports
import random
import typing
import argparse

# local imports
# from src.play import PlayCommand
from src.utils.model_utils import ModelUtils

# define the opponent PYTHON type
opponent_tech_t = typing.Literal["human", "stockfish", "chessbotml"]
color_t = typing.Literal["white", "black"]

# stdlib imports
import os
import typing

# pip imports
import chess
import chess.polyglot
from stockfish import Stockfish
import dotenv


# local imports
from src.encoding.board_encoding import BoardEncoding
from src.utils.model_utils import ModelUtils
from src.utils.uci2class_utils import Uci2ClassUtils
from src.libs.chess_player import ChessPlayer
from src.utils.pgn_utils import PGNUtils
from src.utils.termcolor_utils import TermcolorUtils
from src.libs.chess_extra import ChessExtra
from src.libs.types import opponent_tech_t, color_t


# Init dotenv to load environment variables from .env file
dotenv.load_dotenv()

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.join(__dirname__, "..", "output")
model_folder_path = os.path.join(output_folder_path, "model")
data_folder_path = os.path.join(__dirname__, "..", "data")


class PlayCommand:

    ###############################################################################
    ###############################################################################
    # 	 Play a game of chess between ChessBotML and an opponent (human or Stockfish).
    ###############################################################################
    ###############################################################################

    @staticmethod
    def play_game(
        model_name: str,
        chatbotml_color: color_t = "white",
        opponent_tech: opponent_tech_t = "stockfish",
        stockfish_elo: int = 1350,
        stockfish_depth: int = 10,
        max_ply: int = 200,
    ):
        """
        Play a game of chess between ChessBotML and an opponent (human or Stockfish).

        Parameters:
        - chessbotml_color (color_t): The color for ChessBotML ("white" or "black").
        - opponent_tech (opponent_tech_t): The type of opponent ("human" or "stockfish").
        """

        ###############################################################################
        # Load the Model & mapping and Move to GPU if Available
        #

        # Load the model
        model = ModelUtils.load_model(model_name, model_folder_path)

        # Read the polyglot opening book
        polyglot_path = os.path.join(data_folder_path, "./polyglot/lichess_pro_books/lpb-allbook.bin")
        polyglot_reader = chess.polyglot.open_reader(polyglot_path)

        # Initialize the Chess Player with the loaded model
        chess_player = ChessPlayer(model=model, color=chess.WHITE, polyglot_reader=polyglot_reader)

        ###############################################################################
        # Use the ```predict_move``` function to get the best move and its probabilities for a given board state:
        #

        # Initialize a chess board
        board = chess.Board()

        # display the initial board
        print(ChessExtra.board_to_string(board, flip_board=False if chatbotml_color == "white" else True))

        print(f'White: {TermcolorUtils.cyan("chessbotml" if chatbotml_color == "white" else opponent_tech)}')
        print(f'Black: {TermcolorUtils.cyan("chessbotml" if chatbotml_color == "black" else opponent_tech)}')

        # get stockfish path from environment variable
        stockfish_path = os.getenv("STOCKFISH_PATH")
        assert stockfish_path is not None, "STOCKFISH_PATH environment variable is not set. Please set it in the .env file."

        stockfish_evaluation = Stockfish(path=stockfish_path)
        # stockfish_evaluation.set_depth(20)
        # stockfish_evaluation.set_elo_rating(20)

        # Initialize Stockfish if needed
        if opponent_tech == "stockfish":
            stockfish = Stockfish(path=stockfish_path)
            stockfish.set_elo_rating(stockfish_elo)
            stockfish.set_depth(stockfish_depth)

        ###############################################################################
        #   Play the game
        #
        while True:
            turn_color = "white" if board.turn == chess.WHITE else "black"

            # display the separator between boards
            print(TermcolorUtils.magenta("-" * 50))

            ###############################################################################
            #   Decide the move to play
            #

            # decide who is to play
            player_type = "chessbotml" if chatbotml_color == turn_color else opponent_tech

            # predict the move depending who is it to play. chessbotml or opponent (human or stockfish)
            if player_type == "chessbotml":
                # predict the best move
                best_move = chess_player.predict_next_move(board)

                if best_move is None:
                    # raise an error if no legal moves are available...
                    # as it a ML error, as chess module didnt declare the game over
                    raise ValueError("No legal moves available. Game over.")

                # show the board
                # print(f"Predicted Move {board.fullmove_number} for {turn_color}: {best_move} ")
            elif player_type == "human":
                # get the move from the human player
                legal_moves = [move.uci() for move in board.legal_moves]
                print(f"Legal Moves: {legal_moves}")
                best_move = input(f"Enter your move {board.fullmove_number} for {turn_color} (in UCI format): ")
                while best_move not in legal_moves:
                    print("Invalid move. Please enter a legal move in UCI format.")
                    best_move = input(f"Enter your move {board.fullmove_number} for {turn_color} (in UCI format): ")
            elif player_type == "stockfish":
                # Set the current board position in Stockfish
                stockfish.set_fen_position(board.fen())  # type: ignore

                # Get the best move from Stockfish
                best_move = stockfish.get_best_move()  # type: ignore
                if best_move is None:
                    raise ValueError("Stockfish could not find a move. BUG BUG.")

                # print(f"Stockfish Move {board.fullmove_number} for {turn_color}: {best_move} ")
            else:
                raise NotImplementedError("Only human opponent is implemented for now.")

            ###############################################################################
            #   Make the move on the board
            #

            # Make the move on the board
            board.push_uci(best_move)

            # display the post-move board
            # print(board.unicode())
            print(ChessExtra.board_to_string(board, flip_board=False if chatbotml_color == "white" else True))

            # display the post-move board
            in_opening_str = " (in opening book)" if ChessExtra.in_opening_book(board, polyglot_reader) else ""
            in_endgame_str = " (in endgame)" if ChessExtra.is_endgame(board) else ""
            print(f"Move {board.fullmove_number} played by {turn_color} ({player_type}): {TermcolorUtils.cyan(best_move)} {in_opening_str}{in_endgame_str}")

            ###############################################################################
            #   Optionally, evaluate the position using Stockfish after each move
            #
            position_eval_enabled = True
            if position_eval_enabled is True:
                # evaluate the resulting position and display it
                stockfish_evaluation.set_fen_position(board.fen())
                evaluation = stockfish_evaluation.get_evaluation()
                eval_value = evaluation["value"] if chatbotml_color == "white" else -evaluation["value"]

                eval_str = f"{eval_value/100}" if evaluation["type"] == "cp" else f"mate in {eval_value}"
                print(f"Stockfish evaluation (white pov): {TermcolorUtils.cyan(eval_str)}")

            # Check for game over
            if board.is_game_over():
                game_outcome = typing.cast(chess.Outcome, board.outcome())
                outcome_str = "1-0" if game_outcome.winner is True else "0-1" if game_outcome.winner is False else "1/2-1/2"
                print(f"Game Over! outcome: {TermcolorUtils.cyan(outcome_str)}")
                break

            if board.fullmove_number * 2 >= max_ply:
                print(f"Maximum number of ply ({max_ply}) reached. Declaring the game a draw.")
                break

        ###############################################################################
        # Print the PGN representation of the game, at the end of the game
        #

        # display the separator between boards
        print(TermcolorUtils.magenta("-" * 50))

        opponent_name = "Human" if opponent_tech == "human" else f"Stockfish {stockfish_elo}"
        white_player = "ChessBotML" if chatbotml_color == "white" else opponent_name
        black_player = "ChessBotML" if chatbotml_color == "black" else opponent_name
        pgn_game = PGNUtils.board_to_pgn(board, white_player=white_player, black_player=black_player)

        print("PGN Representation of the game:")
        print(pgn_game)


###############################################################################
###############################################################################
# 	 Main Entry Point
###############################################################################
###############################################################################

if __name__ == "__main__":

    ###############################################################################
    #   Parse command line arguments
    #
    argParser = argparse.ArgumentParser(
        description="Play a game of chess between ChessBotML and an opponent (human or Stockfish).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argParser.add_argument(
        "--color",
        "-c",
        type=str,
        choices=["white", "black", "random"],
        default="random",
        help="The color for ChessBotML (white or black). Default is random.",
    )
    argParser.add_argument(
        "--opponent",
        "-o",
        type=str,
        choices=["human", "stockfish"],
        default="stockfish",
        help="The type of opponent (human or stockfish). Default is stockfish.",
    )
    argParser.add_argument(
        "--stockfish-elo",
        "-e",
        "-elo",
        type=int,
        default=10,
        help="The ELO rating for Stockfish (if opponent is stockfish). Default is 1350.",
    )
    argParser.add_argument(
        "--stockfish-depth",
        "-D",
        type=int,
        default=1,
        help="The search depth for Stockfish (if opponent is stockfish). Default is 10.",
    )
    argParser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode.",
    )
    argParser.add_argument(
        "--model_name",
        "-mn",
        type=str,
        default=ModelUtils.MODEL_NAME.CHESS_MODEL_CONV2D,
        choices=ModelUtils.get_supported_models(),
        help="Model architecture to use for playing",
    )
    argParser.add_argument(
        "--max-ply",
        "-mp",
        type=int,
        default=200,
        help="Maximum number of ply (half-moves) for the game. Default is 200.",
    )
    args = argParser.parse_args()

    if args.debug is True:
        print(f"Arguments: {args}")

    #  decide the color for chessbotml
    if args.color == "random":
        chessbotml_color: color_t = random.choice(["white", "black"])
    else:
        chessbotml_color: color_t = args.color
    opponent_tech: opponent_tech_t = args.opponent

    ###############################################################################
    #   Start the game
    #
    PlayCommand.play_game(
        model_name=args.model_name,
        chatbotml_color=chessbotml_color,
        opponent_tech=opponent_tech,
        stockfish_elo=args.stockfish_elo,
        stockfish_depth=args.stockfish_depth,
        max_ply=args.max_ply,
    )
