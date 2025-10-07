#!/usr/bin/env python3

# stdlib imports
import random
import typing
import argparse

# local imports
from src.play import PlayCommand

# define the opponent PYTHON type
opponent_tech_t = typing.Literal["human", "stockfish", "chessbotml"]
color_t = typing.Literal["white", "black"]

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
        default="ChessModelConv2d",
        choices=["ChessModelConv2d", "ChessModelResNet", "AlphaZeroNet"],
        help="Model architecture to use for playing",
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
    )
