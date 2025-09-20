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
from libs.io_utils import IOUtils
from libs.pgn_utils import PGNUtils
from libs.utils import Utils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = f"{__dirname__}/../output/"

# define the opponent PYTHON type
opponent_tech_t = typing.Literal["human", "stockfish", "chessbotml"]
color_t = typing.Literal["white", "black"]


def play_game(chatbotml_color: color_t = "white", opponent_tech: opponent_tech_t = "stockfish", stockfish_elo: int = 1350, stockfish_depth: int = 10):
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
    print(f"Using device: {device}")

    # move the model to the device and set it to eval mode
    model.to(device)

    # create the reverse mapping
    classindex_to_uci: dict[int, str] = {v: k for k, v in uci_to_classindex.items()}

    ###############################################################################
    # Use the ```predict_move``` function to get the best move and its probabilities for a given board state:
    #

    # Initialize a chess board
    board = chess.Board()
    print(board.unicode())
    print("\n")

    # Initialize Stockfish if needed
    if opponent_tech == "stockfish":
        stockfish_path = "/Users/jetienne/Downloads/stockfish/stockfish-macos-m1-apple-silicon"  # Update this path to your Stockfish binary
        stockfish = Stockfish(path=stockfish_path)
        stockfish.set_elo_rating(stockfish_elo)
        stockfish.set_depth(stockfish_depth)

    while True:
        turn_color = "white" if board.turn == chess.WHITE else "black"

        if chessbotml_color == turn_color:
            # predict the best move
            best_move = Utils.predict_next_move(board, model, device, classindex_to_uci)

            if best_move is None:
                # raise an error if no legal moves are available...
                # as it a ML error, as chess module didnt declare the game over
                raise ValueError("No legal moves available. Game over.")

            # show the board
            print(f"Predicted Move {board.fullmove_number} for {turn_color}: {best_move} ")
        elif opponent_tech == "human":
            # get the move from the human player
            legal_moves = [move.uci() for move in board.legal_moves]
            print(f"Legal Moves: {legal_moves}")
            best_move = input(f"Enter your move {board.fullmove_number} for {turn_color} (in UCI format): ")
            while best_move not in legal_moves:
                print("Invalid move. Please enter a legal move in UCI format.")
                best_move = input(f"Enter your move {board.fullmove_number} for {turn_color} (in UCI format): ")
        elif opponent_tech == "stockfish":
            # Set the current board position in Stockfish
            stockfish.set_fen_position(board.fen())

            # Get the best move from Stockfish
            best_move = stockfish.get_best_move()
            if best_move is None:
                raise ValueError("Stockfish could not find a move. BUG BUG.")

            print(f"Stockfish Move {board.fullmove_number} for {turn_color}: {best_move} ")
        else:
            raise NotImplementedError("Only human opponent is implemented for now.")

        # Make the move on the board
        board.push_uci(best_move)
        print(board.unicode())
        print("\n")

        if board.is_game_over():
            print(f"Game Over! outcome: {board.outcome()}")
            break

    ###############################################################################
    # Optionally, print the PGN representation of the game

    print("PGN Representation of the game:")

    opponent_name = "Human" if opponent_tech == "human" else f"Stockfish {stockfish_elo}"

    white_player = "ChessBotML" if chessbotml_color == "white" else opponent_name
    black_player = "ChessBotML" if chessbotml_color == "black" else opponent_name

    pgn_game = PGNUtils.board_to_pgn(board, white_player=white_player, black_player=black_player)
    print(pgn_game)


###############################################################################
###############################################################################
# 	 Main Entry Point
###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Play a game of chess between ChessBotML and an opponent (human or Stockfish)."
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["white", "black", "random"],
        default="random",
        help="The color for ChessBotML (white or black). Default is random.",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["human", "stockfish"],
        default="stockfish",
        help="The type of opponent (human or stockfish). Default is stockfish.",
    )
    parser.add_argument(
        "--stockfish-elo",
        type=int,
        default=10,
        help="The ELO rating for Stockfish (if opponent is stockfish). Default is 1350.",
    )
    parser.add_argument(
        "--stockfish-depth",
        type=int,
        default=1,
        help="The search depth for Stockfish (if opponent is stockfish). Default is 10.",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode.",
    )
    args = parser.parse_args()

    if args.debug is True:
        print(f"Arguments: {args}")

    # Play the game with the specified arguments
    if args.color == "random":
        chessbotml_color: color_t = random.choice(["white", "black"])
    else:
        chessbotml_color: color_t = args.color
    opponent_tech: opponent_tech_t = args.opponent
    play_game(chatbotml_color=chessbotml_color, opponent_tech=opponent_tech, stockfish_elo=args.stockfish_elo, stockfish_depth=args.stockfish_depth)
