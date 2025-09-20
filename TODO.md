# TODO
- evaluate the model on a validation set during training
- do a `./play.py` able to play against a human and stockfish, with good cmdline args
  - cmdline options:
    - `--model-path` to load a model
    - `--engine-path` to load a chess engine (stockfish)
    - `--time` time per move for the engine
    - `--color` white or black for the ml bot
  - display the board in ascii
  - move in uci format (e2e4)
  - display the move suggested by the model and the move suggested by stockfish

# DONE
- DONE that a test dataset is separate from the training dataset

# Ideas

## to have multiple models for each stage of the game (opening, midgame, endgame)
The model is very strong in the opening, i think it is due to the openings are standard so there not much variability. 
We can leverage the same principles in the endgame as the number of pieces is lower.

- Q. how to detect the stage of the game?
- A. by the number of pieces left on the board
- usually we say endgame is when there is no more queen on the board
- opening is the first 10-15 moves... or can be detected with a database of openings

## to have multiple models for each color (white, black) ?
Seems not necessary, the model can learn the difference of playing white or black from the board state.

## Do a validation set during training
- pytorch validation doc
  - https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html

## Link it with a chess engine to play games
see `./test_stockfish.py`

- https://www.freecodecamp.org/news/create-a-self-playing-ai-chess-engine-from-scratch/
- https://github.com/jackdawkins11/pytorch-alpha-zero

## Change the encoding of the board
- gym-chess got exactly alphazero encoding https://pypi.org/project/gym-chess/#chessalphazero-v0

- https://jdhwilkins.com/1-hot-encoding-the-chess-programmers-secret-weapon/
- https://jdhwilkins.com/python-chess-efficient-move-generation-using-bitwise-operations/
- https://jdhwilkins.com/python-chess-game-simulation-and-illegal-moves/
- https://www.chessprogramming.org/

- https://www.freecodecamp.org/news/create-a-self-playing-ai-chess-engine-from-scratch/
- about alphazero encoding:
  - https://blog.devgenius.io/creating-an-ai-chess-engine-part-2-encoding-using-the-alphazero-method-63c3c3c3a960

Current board encoding:
- 8x8x12 binary tensor (6 first for white pieces, 6 last for black pieces)
  - 6 because each color got 6 types of pieces (pawn, knight, bishop, rook, queen, king)
  - 1 if piece is present, 0 otherwise
- 8x8x1 binary tensor for the destination of legal moves
- nothing for the source of legal moves <- DO THIS - this is encoding the turn
- nothing for castling rights
- nothing for en-passant

- Q. where it is coded to know the color to play? BUG BUG BUG

- 1x64 binary tensor for the side to play (1 if white to play, 0 if black to play)