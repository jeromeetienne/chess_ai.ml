# chess-bot.ml


A chess bot using machine learning to predict the best move.
Coded with Python and PyTorch.

It has been originally inspired by https://github.com/Skripkon/chess-engine.git

# Machine Learning insights

> "A neural network model on its own cannot produce a reasonable move, as it lacks the depth of calculation required."

## Explaination
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

## Good Looking!

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

The dataset is taken from here:https://database.nikonoel.fr/ and put it in `./data/pgn/`

To train a new model, run:

```bash
python ./src/train.py
```

To predict with the trained model, run:

```bash
python ./src/predict.py
```

## Useful Links
- stackoverflow specific to chess: http://chess.stackexchange.com/
- alpha zero board/move encoding - https://github.com/iamlucaswolf/gym-chess/blob/master/gym_chess/
  - [board_encoding.py](https://github.com/iamlucaswolf/gym-chess/blob/master/gym_chess/alphazero/board_encoding.py)
  - [move_encoding](https://github.com/iamlucaswolf/gym-chess/tree/master/gym_chess/alphazero/move_encoding)
- [AlphaZero paper](https://arxiv.org/abs/1712.01815)
- [AlphaZero paper2](https://arxiv.org/abs/2304.14918)
- [alpha vile](https://www.informatik.tu-darmstadt.de/fb20/aktuelles_fb20/fb20_news/news_fb20_details_308928.en.jsp) - the second version of alpha zero
- [article](https://ar5iv.labs.arxiv.org/html/2304.14918) explaining why transformer may be unsuitable for chess
- chess.com article [Another Silly Question: How Many Chess Moves Are There?](https://www.chess.com/blog/the_real_greco/another-silly-question-how-many-chess-moves-are-there)