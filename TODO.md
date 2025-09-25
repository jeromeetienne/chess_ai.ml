# TODO
- generalize the game slice in the dataset builder
  - allow not to set begining and end move_index
  1. build a dataset for each stage of the game (opening, midgame, endgame)
  2. train a model for each stage of the game
  3. during play, detect the stage of the game and use the corresponding model
- code a way to train on a special range of moves, not the whole game
  - select by move number (e.g. 10 to 30)
  - later by dynamically detecting opening, midgame, endgame
  - generate multiple dataset files for each stage of the game
- look for model structure on the web
  - search for 'pytorch chess model'
  - search for 'pytorch chess neural network'
  - search for 'pytorch/tensorflow alpha zero github'
- make it play on lichess ?
- organize `./libs`

# DONE
- DONE read opening books
  - here are some pgn https://sites.google.com/site/computerschess/download
  - more polyglot opening books https://github.com/michaeldv/donna_opening_books/
  - several collection https://chess.stackexchange.com/questions/35448/looking-for-polyglot-opening-books
  - how to use it https://chess.stackexchange.com/questions/24738/how-to-use-opening-books-on-mac-linux
- DONE implement better looking board display - between ascii art and color - it is possible to do something more readable
  - https://rebel13.nl/download/books.html
- WONTDO plug stockfish into python chess
  - https://python-chess.readthedocs.io/en/latest/engine.html
- DONE do early stopping during training
- DONE evaluate the model on a validation set during training
- DONE add proper logs in `./train.py` and `./predict.py`
- DONE connect `./play.py` to lichess.org to play online
  - add it in `Makefile`
- DONE in `./train.py`, save the model every N epochs
- DONE do a `./play.py` able to play against a human and stockfish, with good cmdline args
  - cmdline options:
    - `--model-path` to load a model
    - `--engine-path` to load a chess engine (stockfish)
    - `--time` time per move for the engine
    - `--color` white or black for the ml bot
  - display the board in ascii
  - move in uci format (e2e4)
  - display the move suggested by the model and the move suggested by stockfish
- DONE in `./train.py`, cache the dataset in numpy format for faster loading
- DONE clean up `./train.py` and `./predict.py`
- DONE rename X to `board_tensor`, y to `move_tensor`
  - or `input_tensors`, `expected_target_tensors`
- DONE rename `move_to_int` to `uci_to_classindex`
- DONE that a test dataset is separate from the training dataset
- DONE display chess board with unicode characters
