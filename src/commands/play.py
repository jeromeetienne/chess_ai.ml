# stdlib imports
import os
import random
import typing
import argparse

# pip imports
import torch
import chess
from stockfish import Stockfish

# local imports
from ..libs.chessbotml_player import ChessbotMLPlayer
from ..libs.io_utils import IOUtils
from ..libs.pgn_utils import PGNUtils
from ..libs.termcolor_utils import TermcolorUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = f"{__dirname__}/../../output/"

# define the opponent PYTHON type
opponent_tech_t = typing.Literal["human", "stockfish", "chessbotml"]
color_t = typing.Literal["white", "black"]

class PlayCommand:

    ###############################################################################
    ###############################################################################
    # 	 Play a game of chess between ChessBotML and an opponent (human or Stockfish).
    ###############################################################################
    ###############################################################################

    @staticmethod   
    def play_game(
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

        # Load the mapping
        uci_to_classindex = IOUtils.load_uci_to_classindex(folder_path=output_folder_path)
        num_classes = len(uci_to_classindex)

        # Load the model
        model = IOUtils.load_model(folder_path=output_folder_path, num_classes=num_classes)

        # Check for GPU
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
        # print(f"Using device: {device}")

        # move the model to the device and set it to eval mode
        model.to(device)

        # create the reverse mapping
        classindex_to_uci: dict[int, str] = {v: k for k, v in uci_to_classindex.items()}

        chatbotml_player = ChessbotMLPlayer(model=model, classindex_to_uci=classindex_to_uci)

        ###############################################################################
        # Use the ```predict_move``` function to get the best move and its probabilities for a given board state:
        #

        # Initialize a chess board
        board = chess.Board()
        print(board.unicode())

        print(f'White: {TermcolorUtils.cyan("chessbotml" if chatbotml_color == "white" else opponent_tech)}')
        print(f'Black: {TermcolorUtils.cyan("chessbotml" if chatbotml_color == "black" else opponent_tech)}')

        stockfish_path = "/Users/jetienne/Downloads/stockfish/stockfish-macos-m1-apple-silicon"  # Update this path to your Stockfish binary
        stockfish_evaluation = Stockfish(path=stockfish_path)
        # stockfish_evaluation.set_depth(20)
        stockfish_evaluation.set_elo_rating(2000)

        # Initialize Stockfish if needed
        if opponent_tech == "stockfish":
            stockfish_path = "/Users/jetienne/Downloads/stockfish/stockfish-macos-m1-apple-silicon"  # Update this path to your Stockfish binary
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
                best_move = chatbotml_player.predict_next_move(board)

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
                stockfish.set_fen_position(board.fen())

                # Get the best move from Stockfish
                best_move = stockfish.get_best_move()
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
            print(board.unicode())

            # display the post-move board
            print(f"Move {board.fullmove_number} played by {turn_color} ({player_type}): {TermcolorUtils.cyan(best_move)} ")
            # print(f'in opening book: {chatbotml_player.is_in_opening_book(board)}')
            print(f'in opening book: {TermcolorUtils.green("yes") if chatbotml_player.is_in_opening_book(board) else "no"}')

            ###############################################################################
            #   Optionally, evaluate the position using Stockfish after each move
            #
            position_eval_enabled = True
            if position_eval_enabled is True:
                # evaluate the resulting position and display it
                stockfish_evaluation.set_fen_position(board.fen())
                evaluation = stockfish_evaluation.get_evaluation()
                eval_value = evaluation["value"]
                eval_str = f"{eval_value/100}" if evaluation["type"] == "cp" else f"mate in {eval_value}"
                print(f"Stockfish evaluation: {TermcolorUtils.cyan(eval_str)} for white")

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

