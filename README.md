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
