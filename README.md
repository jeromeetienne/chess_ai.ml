# chess-bot.ml


A chess bot using machine learning to predict the best move.
Coded with Python and PyTorch.

It has been originally inspired by https://github.com/Skripkon/chess-engine.git

## Inputs encoding
The input is a 8x8x14 tensor representing the chess board. 

Each of the 12 channels corresponds to a piece type (6 for white, 6 for black). 
A value of 1 in a channel indicates the presence of the corresponding piece on that square, while a value of 0 indicates its absence.

The remaining 2 channels are used to encode the legal moves:
- The 13th channel encodes the **from** square of the move (1 if the piece is to be moved from that square, 0 otherwise).
- The 14th channel encodes the **to** square of the move (1 if the piece is to be moved to that square, 0 otherwise).

## Output encoding
The output is a 4096-dimensional vector representing all possible moves on a chessboard based on the dataset.

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