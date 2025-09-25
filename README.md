# chess-bot.ml

heavyly inspired by https://github.com/Skripkon/chess-engine.git

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
