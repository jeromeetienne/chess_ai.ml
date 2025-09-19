# TODO
- evaluate the model on a validation set during training
- that a test dataset is separate from the training dataset

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

## Change the encoding of the board
- https://jdhwilkins.com/1-hot-encoding-the-chess-programmers-secret-weapon/
- https://jdhwilkins.com/python-chess-efficient-move-generation-using-bitwise-operations/
- https://jdhwilkins.com/python-chess-game-simulation-and-illegal-moves/
- https://www.chessprogramming.org/

Current board encoding:
- 8x8x12 binary tensor (6 first for white pieces, 6 last for black pieces)
  - 6 because each color got 6 types of pieces (pawn, knight, bishop, rook, queen, king)
  - 1 if piece is present, 0 otherwise
- 8x8x1 binary tensor for the destination of legal moves
- nothing for the source of legal moves
- nothing for castling rights
- nothing for en-passant

- Q. where it is coded to know the color to play? BUG BUG BUG

- 1x64 binary tensor for the side to play (1 if white to play, 0 if black to play)