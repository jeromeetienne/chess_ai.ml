"""
Module where you pick the move encoding strategy for the whole project.

- To use AlphaZero move encoding, uncomment the first import line.
- To use UCI2Class move encoding, uncomment the second import line.

WARNING: when you change the move encoding here, you must also do the following to avoid inconsistencies:
- delete all previously generated tensors.
- delete all previously trained models.
"""

# Enable this line to use AlphaZero move encoding
from .move_encoding_alphazero import MoveEncodingAlphaZero as MoveEncoding

# Enable this line to use UCI2Class move encoding
# from .move_encoding_uci2class import MoveEncodingUci2Class as MoveEncoding
