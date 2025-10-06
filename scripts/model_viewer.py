import torchview

# local imports
from src.libs.encoding import Encoding
from src.utils.model_utils import ModelUtils

import os

__dirname__ = os.path.dirname(os.path.abspath(__file__))
model_folder_path = os.path.join(__dirname__, "../output/model/")

input_shape = Encoding.get_input_shape()
output_shape = Encoding.get_output_shape()

model = ModelUtils.load_model(folder_path=model_folder_path, input_shape=input_shape, output_shape=output_shape)

# Input size for the model visualization (batch size 1, 21 channels, 8x8 image)
input_size = (1, *input_shape)  # (1, 21, 8, 8)


# Generate the model graph visualization with expanded nested modules
dst_folder = os.path.join(__dirname__, "../")
model_graph = torchview.draw_graph(model, input_size=input_size, expand_nested=True, filename='chess_model', directory=dst_folder)

# # # Display the visual graph (in Jupyter, this will show the image)
visual = model_graph.visual_graph
visual.render("chess_model_graph", format="png")  # Saves and opens the graph image as PNG
