## to have multiple models for each stage of the game (opening, midgame, endgame)
The model is very strong in the opening, i think it is due to the openings are standard so there not much variability. 
We can leverage the same principles in the endgame as the number of pieces is lower.

- https://www.chessprogramming.org/Game_Phases
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

## change encoding of the output best_move
- 2 chessboards one with the source square, one with the destination square
  - 64 squares each
  - 64 + 64 = 128 output neurons
  - much less than the current 1869 output neurons... 
  - currently it is a classification problem with 1869 classes... with only one is correct... poorly formulated problem
- https://www.baeldung.com/cs/encode-chess-board-state
## Learn only winning moves
- ISSUE: currently i learn from any move
- use stockfish to evaluate the position after each move... and use that to filter the dataset
- learn only from the winning side ?
  - what if the looser made great moves during the whole game, and blunder at the end

## use stockfish evaluations
- stockfish is able to evaluate a position with a score in centipawns
- use that to weight the loss function during training ?

# use transformer model instead of convolutional neural network
- https://github.com/Atenrev/chessformers
- transformer and attention mechanism can be used for learning spatial relationships
- "spatial transformer with self attention mechanism on mnist with pytorch" with perplexity

## Change the encoding of the board
- currently it contains only the current pieces positions... with a 0 or 1 per int
  - what about i add the threaten squares? board.attacks()

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