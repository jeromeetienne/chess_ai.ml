# chess-bot.ml

heavyly inspired by https://github.com/Skripkon/chess-engine.git

## Installation

We recommend to use a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

Then install the required packages:

```bash
pip install -e .
```
## How to use

To train a new model, run:

```bash
python ./src/train.py
```

To predict with the trained model, run:

```bash
python ./src/predict.py
```
