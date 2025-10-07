# stdlib imports
import os
import typing

# pip imports
import chess
import chess.polyglot
from stockfish import Stockfish


# local imports
from src.libs.encoding import Encoding
from src.utils.model_utils import ModelUtils
from src.utils.uci2class_utils import Uci2ClassUtils
from .libs.chess_player import ChessPlayer
from .utils.pgn_utils import PGNUtils
from .utils.termcolor_utils import TermcolorUtils
from .libs.chess_extra import ChessExtra
from .libs.types import opponent_tech_t, color_t

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
        model = ModelUtils.create_model(model_name)
        ModelUtils.load_weights(model, model_folder_path)
        
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

        # TODO change that to a ENV variable in python-dotenv
        stockfish_path = "/Users/jetienne/Downloads/stockfish/stockfish-macos-m1-apple-silicon"  # Update this path to your Stockfish binary
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
                stockfish.set_fen_position(board.fen()) # type: ignore

                # Get the best move from Stockfish
                best_move = stockfish.get_best_move() # type: ignore
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
            in_opening_str = " (in opening book)" if ChessExtra.is_in_opening_book(board, polyglot_reader) else ""
            print(f"Move {board.fullmove_number} played by {turn_color} ({player_type}): {TermcolorUtils.cyan(best_move)} {in_opening_str}")

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

