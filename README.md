# chess-bot.ml

A chess bot using machine learning to predict the best move.
Coded with Python and PyTorch.

it includes a UCI engine interface to connect to chess GUIs like Arena or CuteChess.

## Machine Learning insights

> "A neural network model on its own cannot produce a reasonable move, as it lacks the depth of calculation required."

### Explaination

A neural network model **alone** doesn't possess the depth of calculation required in chess. In chess terminology, calculation refers to the process of visualizing sequences of moves and countermoves, assessing their consequences, and predicting resulting positions to make the best decision. This depth of calculation is achieved using search algorithms such as [minimax](https://en.wikipedia.org/wiki/Minimax), [alpha-beta pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning), or [Monte Carlo Tree Search (MCTS)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search), which systematically explore the game tree to a certain depth.

AlphaZero exemplifies the synergy between machine learning and traditional search algorithms. It combines a deep neural network—which evaluates board positions and suggests promising moves, using Monte Carlo Tree Search (MCTS), a powerful tree search technique. The neural network guides the search by focusing on the most relevant moves, while MCTS provides the depth of calculation needed to explore possible continuations. This integration allows AlphaZero to efficiently balance intuition (from ML) and calculation (from tree search), achieving superhuman performance in chess.

## Inputs encoding

The input is an 8x8x21 tensor (21 planes × 8 × 8), matching `Encoding.get_input_shape()` in the code.
The board is always represented from the perspective of the active player (the player to move). If the active player is black, the board is flipped so that black pieces are on the bottom and white pieces are on the top. This allows the model to consistently learn patterns from the viewpoint of the player to move, simplifying the learning process and improving generalization across different game states.

The planes are used to represent piece presence and game metadata. They are (index: name):

- 0: ACTIVE_PAWN
- 1: ACTIVE_KNIGHT
- 2: ACTIVE_BISHOP
- 3: ACTIVE_ROOK
- 4: ACTIVE_QUEEN
- 5: ACTIVE_KING
- 6: OPPONENT_PAWN
- 7: OPPONENT_KNIGHT
- 8: OPPONENT_BISHOP
- 9: OPPONENT_ROOK
- 10: OPPONENT_QUEEN
- 11: OPPONENT_KING
- 12: REPETITION_2
- 13: REPETITION_3
- 14: TURN
- 15: ACTIVE_KINGSIDE_CASTLING_RIGHTS
- 16: ACTIVE_QUEENSIDE_CASTLING_RIGHTS
- 17: OPPONENT_KINGSIDE_CASTLING_RIGHTS
- 18: OPPONENT_QUEENSIDE_CASTLING_RIGHTS
- 19: HALFMOVE_CLOCK
- 20: FULLMOVE_NUMBER

Notes:

- The first 12 planes encode piece occupancy (6 piece types for the active side followed by 6 for the opponent) rather than a fixed white/black ordering; the board is flipped when black is to move so the tensor is always in the perspective of the active player.
- There are no dedicated "from"/"to" planes in this repository's encoding. Moves are encoded as a single class index (see "Output encoding" below).

## Output encoding

The output is a 1968-dimensional vector (one scalar class per possible move) representing the move policy. The exact set and ordering of moves is provided by the UCI→class JSON files in `output/uci2classes/` (for example `uci2class_arr_white.json`).

This is derived at runtime by `Encoding.get_output_shape()` which loads the UCI-to-class mapping (see `src/utils/uci2class_utils.py`). The repository's current mapping file for white contains 1968 entries, so the model outputs logits of shape `(N, 1968)`.

The `_white` and `_black` suffixes in the UCI-to-class mapping filenames indicate separate mappings for white and black to account for board symmetry. When the active player is black, the board is flipped before encoding, and the corresponding black mapping file is used to ensure moves are consistently represented from the active player's perspective.

## Good Looking

Superbe board display with unicode characters
![Ascii chess board](https://github.com/user-attachments/assets/3df3d359-f05f-4cac-8f9a-fcbf9489c985)

## Installation

We recommend to use a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Then install the required packages:

```bash
pip install -e .
```

The dataset is taken from here:<https://database.nikonoel.fr/> and put it in `./data/pgn/`

To train a new model, run:

```bash
python ./src/train.py
```

To play with the trained model, run:

```bash
python ./src/play.py
```

## Credits

- 🎯 **Original Inspiration:** This project draws inspiration from [Skripkon's chess-engine](https://github.com/Skripkon/chess-engine.git) — thank you for paving the way!
- 🐍 **Core Library:** Built on top of the fantastic [`python-chess`](https://python-chess.readthedocs.io/en/latest/) library, which makes chess programming a breeze.
- 🤖 **AlphaZero Team:** Huge kudos to the AlphaZero teams for their groundbreaking research and open inspiration!
- 🙏 **Community:** Thanks to all contributors, testers, and the open-source chess community for sharing knowledge and resources.

## Useful Links

- stackoverflow specific to chess: <http://chess.stackexchange.com/>
- alpha zero board/move encoding - <https://github.com/iamlucaswolf/gym-chess/blob/master/gym_chess/>
  - [board_encoding.py](https://github.com/iamlucaswolf/gym-chess/blob/master/gym_chess/alphazero/board_encoding.py)
  - [move_encoding](https://github.com/iamlucaswolf/gym-chess/tree/master/gym_chess/alphazero/move_encoding)
- [AlphaZero paper](https://arxiv.org/abs/1712.01815)
- [AlphaZero paper2](https://arxiv.org/abs/2304.14918)
- [alpha vile](https://www.informatik.tu-darmstadt.de/fb20/aktuelles_fb20/fb20_news/news_fb20_details_308928.en.jsp) - the second version of alpha zero
- [article](https://ar5iv.labs.arxiv.org/html/2304.14918) explaining why transformer may be unsuitable for chess
- chess.com article [Another Silly Question: How Many Chess Moves Are There?](https://www.chess.com/blog/the_real_greco/another-silly-question-how-many-chess-moves-are-there)

## Insight 2

- "Look closely at your data—model problems often stem from hidden biases or imbalances in the dataset, not just the model itself."

### Explanation

im running a dual head model which predicts both the move and the game outcome (win/draw/loss) from a given position, so it does classification and regression
at the same time. Unfortunatly the regression head is not learning efficiently, and the loss was varying widely.
I tried to fix it with various strategies:

- normalizing the outcome to be between 0 and 1 with a sigmoid
- using a dynamic weighted loss. Useful to balance change in the classification and regression loss. i was hoping that if the regression loss is
  high, it will be weighted less
- using mean absolute square error instead of mean square error. MAE is less sensitive to outliers.
- using a smaller learning rate for the regression head.

Nothing worked. :(

After some time, independantly, i noticed my dataset was unbalanced. It was games from stockfish and stockfish **never resign**, so some games were very long
(e.g. 100 moves or more). I found that silly and i filtered out endgames from the dataset, using the [speelman criteria](https://www.chess.com/blog/introuble2/the-value-of-the-active-king)
and suddently the regression head was learning much better, and the loss was more stable.

## Machine Learning Tips

- if you use MeanAbsoluteError (`L1Loss`) for regression,
  - consider using `SmoothL1Loss` ([Huber loss](https://en.wikipedia.org/wiki/Huber_loss)) instead. It is less sensitive to outliers and can provide more stable gradients, leading to better convergence during training.
  - it limits the gradient explosion, especially when batches are large.
  - `SmoothL1Loss` got the best of both worlds: it behaves like L2Loss when the error is small (which encourages convergence) and like L1Loss when the error is large (which reduces the influence of outliers).
- [Convolutional Layer](https://en.wikipedia.org/wiki/Convolutional_layer) learns spatial data (like chess boards) much better than undefined
layers (e.g. list of class indexes)
  - consider using Conv2D layers for board representation instead of flattening the board into a 1D vector.
  - **NOTE**: this is the motivation behind AlphaZero output encoding: they picked a much larger representation the output (8x8x73 planes aka 4672 cells) to be able to use Conv2D layers.
  If you picked list of classes, it would have been 1968 classes (All legal move in chess, independantly of the position), which is not spatial anymore.
