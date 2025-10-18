#!/usr/bin/env python3

# stdlib imports
import os

# pip imports
import torchview
import argparse

# local imports
from src.encoding.board_encoding import BoardEncoding
from src.encoding.move_encoding_uci2class import MoveEncodingUci2Class as MoveEncoding
from src.utils.model_utils import ModelUtils


__dirname__ = os.path.dirname(os.path.abspath(__file__))
model_folder_path = os.path.abspath(os.path.join(__dirname__, "../output/model/"))

# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    # Parse command line arguments
    argParser = argparse.ArgumentParser(description="Generate model visualization in a PNG.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argParser.add_argument(
        "--model_name",
        "-mn",
        type=str,
        default=ModelUtils.MODEL_NAME.CHESS_MODEL_CONV2D,
        choices=ModelUtils.get_supported_models(),
        help="Model architecture to use for training",
    )
    args = argParser.parse_args()
    input_shape = BoardEncoding.get_input_shape()
    output_shape = MoveEncoding.get_output_shape()

    # Create the model
    model_name = args.model_name
    model = ModelUtils.create_model(model_name)

    # Input size for the model visualization (batch size 1, 21 channels, 8x8 image)
    input_size = (1, *input_shape)  # (1, 21, 8, 8)

    # Generate the model graph visualization with expanded nested modules
    dst_folder = os.path.join(__dirname__, "../")
    model_graph = torchview.draw_graph(model, input_size=input_size, expand_nested=True, filename="chess_model", directory=dst_folder)

    # # # Display the visual graph (in Jupyter, this will show the image)
    visual = model_graph.visual_graph
    visual.render("chess_model_graph", format="png")  # Saves and opens the graph image as PNG
