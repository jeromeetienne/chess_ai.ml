# stdlib imports
import os

# pip imports
import torch
import chess

# local imports
from .libs.chess_player import ChessPlayer
from .libs.io_utils import IOUtils
from .libs.pgn_utils import PGNUtils
from .libs.encoding import Encoding

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.join(__dirname__, "..", "output")

class PredictCommand:
    @staticmethod
    def predict():
        ###############################################################################
        # 2. Load the Model & mapping and Move to GPU if Available
        #

        # Load the mapping
        uci2class_white = IOUtils.uci2class_load(chess.WHITE)
        num_classes = len(uci2class_white)


        # Load the model
        input_shape = Encoding.INPUT_SHAPE  # (channels, height, width)
        output_shape = (num_classes,)
        model = IOUtils.load_model(folder_path=output_folder_path, input_shape=input_shape, output_shape=output_shape)

        # Check for GPU
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
        print(f"Using device: {device}")

        # move the model to the device and set it to eval mode
        model.to(device)
        model.eval()  # Set the model to evaluation mode (it may be reductant)

        # create the reverse mapping
        classindex_to_uci: dict[int, str] = {v: k for k, v in uci_to_classindex.items()}

        chess_player = ChessPlayer(model=model, classindex_to_uci=classindex_to_uci)

        ###############################################################################
        # 3. Use the ```predict_move``` function to get the best move and its probabilities for a given board state:
        #

        # Initialize a chess board
        board = chess.Board()
        print(board.unicode())
        print("\n")

        while True:
            # predict the best move
            best_move = chess_player.predict_next_move(board)

            if best_move is None:
                # raise an error if no legal moves are available...
                # as it a ML error, as chess module didnt declare the game over
                raise ValueError("No legal moves available. Game over.")

            turn_color = "white" if board.turn == chess.WHITE else "black"

            # show the board
            print(f"Predicted Move {board.fullmove_number} for {turn_color}: {best_move} ")

            # Make the move on the board
            board.push_uci(best_move)
            print(board.unicode())
            print("\n")

            print(f"Board Outcome: {board.outcome()}")
            if board.is_game_over():
                print("Game Over!")
                break

        ###############################################################################
        # Optionally, print the PGN representation of the game

        print("PGN Representation of the game:")
        pgn_game = PGNUtils.board_to_pgn(board)
        print(pgn_game)
