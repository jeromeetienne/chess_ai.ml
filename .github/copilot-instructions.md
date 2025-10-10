## Quick orientation for AI coding agents

This repository is a small ML-based chess engine that learns a move policy from PGN games and integrates with a search/opening book. Below are the essential facts and patterns to be productive.

### Coding style
- Use explicit names for variables and functions.
  - never use single-letter names
  - prefer descriptive names over short names
- Never use tabs; always 4 spaces.
- Follow PEP8 and type hints (enforced by `make lint`).
- Use f-strings for formatting (no `%` or `.format()`).

### Big picture
- Data pipeline: PGN files (data/pgn, data/lichess_elite, data/fishtest_stockfish) -> scripts/pgn_splitter.py -> dataset builders (`bin/build_boards_moves.py`, `bin/build_evals_fishtest.py`) -> tensor files in `output/pgn_tensors/` (see naming in `DatasetUtils.FILE_SUFFIX`).
- Model: PyTorch models live in `src/libs/chess_model.py` (ResNet/Conv variants). Models are saved/loaded via `src/utils/model_utils.py` to `output/model/model.pth`.
- Serving/play: `src/play.py` and `bin/play.py` load the model, use `ChessPlayer` to predict moves and optionally play against Stockfish or a human. A polyglot opening book (data/polyglot/...) is used to skip openings and to provide moves.

### Important files to reference
- Input/output encodings: `src/libs/encoding.py` — input shape is Encoding.get_input_shape() (channels,height,width) and currently uses 21 planes (see `Encoding.PLANE`). Output shape is `Encoding.get_output_shape()` and is derived from `output/uci2classes/` (see `Uci2ClassUtils`).
- Dataset helpers: `src/utils/dataset_utils.py` — load/save tensor conventions and helpers to convert PGN -> tensors. Tensors are saved as `{basename}_boards_tensor.pt`, `{basename}_moves_tensor.pt`, `{basename}_evals_tensor.pt` in `output/pgn_tensors`.
- UCI mapping: `src/utils/uci2class_utils.py` — mappings are loaded from `output/uci2classes/uci2class_arr_white.json` and `..._black.json`. These determine the number of classes (model output width).
- Model utilities: `src/utils/model_utils.py` — summary, save/load model behavior. Saved file: `output/model/model.pth` (torch.state_dict).
- Entrypoints: `src/train.py`, `src/play.py`, `bin/*.py`, `scripts/*` (e.g., `scripts/pgn_splitter.py`).
- Top-level orchestration: `Makefile` includes useful targets (`make train`, `make build_boards_moves`, `make build_evals_fishtest`, `make full_pipeline`, `make lint`).

### Run / developer workflows (concrete commands)
- Setup recommended: Python 3.10 virtualenv + `pip install -e .` (see `pyproject.toml` for pinned deps).
- Lint/typecheck: `make lint` (runs `pyright` on `src/` and `bin/`).
- Build dataset (example):
  - Split PGNs: `./scripts/pgn_splitter.py -d ./output/pgn_splits -v ./data/lichess_elite/*.pgn`
  - Build boards/moves: `./bin/build_boards_moves.py` (writes tensors to `output/pgn_tensors/`).
  - Build evals (from fishtest): `./bin/build_evals_fishtest.py` or slow live stockfish evaluation `./bin/build_evals_stockfish.py`.
- Train: `./bin/train.py` (or `python ./src/train.py`). Default args in `TrainCommand.train()` include large batch sizes (2048) and default max files 15; tests/quick runs should lower these (use `-fc` flags on `bin/train.py`).
- Play/serve: `./bin/play.py` loads `output/model/model.pth` and the polyglot book at `data/polyglot/lichess_pro_books/lpb-allbook.bin`. The path to the Stockfish binary is currently hard-coded in `src/play.py` — update locally before running.

### Project-specific conventions and gotchas
- Encoding plane count is 21 (see `Encoding.PLANE`). Many routines depend on that exact shape — changing it requires updating all places that call `Encoding.get_input_shape()`.
- Move classes are not generated at runtime: `Uci2ClassUtils` expects JSON arrays in `output/uci2classes/`. If you change the move-set or regenerate mappings, you must rebuild datasets and retrain models to keep `model.pth` and UCI mappings consistent.
- Dataset files naming: tensors must follow `{basename_prefix}_boards_tensor.pt`, `{basename_prefix}_moves_tensor.pt`, `{basename_prefix}_evals_tensor.pt`. `DatasetUtils.load_datasets()` concatenates files in `output/pgn_tensors/` in sorted order.
- Model load/save contract: `ModelUtils.load_model()` constructs a `ChessModel(input_shape, output_shape)` and then `load_state_dict()` — architecture must match the saved state. If you alter `chess_model.py` (e.g., channels or final layer size), either update saved model or provide migration code.
- GPU detection and usage: `src/train.py` uses `torch.accelerator.*` API to detect device. Be aware this may differ across PyTorch versions; tests on your environment may require small adjustments (e.g., fallback to `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).

### Debugging tips / common errors
- "Unexpected key / size mismatch" when loading state dict: check that `Encoding.get_output_shape()[0]` matches number of classes in `output/uci2classes/*` and that `ChessModel` final Linear layer width equals that number.
- If DatasetUtils reports different counts for boards/moves/evals, inspect `output/pgn_tensors/` for missing or misnamed files; `DatasetUtils.load_datasets()` will print how many files and positions it loads.
- If play crashes with Stockfish: verify `stockfish` binary path in `src/play.py` or set up compatibility with `stockfish` package on your platform.

### Examples to cite in PRs
- When changing input planes, reference `src/libs/encoding.py` and update `Encoding.PLANE` + `Encoding.get_input_shape()` and all code that assumes 21 planes (train, dataset conversion, model definition).
- When changing move classes, update `scripts/build_uci2class.py` (exists in `scripts/`) and ensure `output/uci2classes/*.json` are regenerated and datasets rebuilt.

If anything here is unclear or you want more detail (argument lists for `bin/*.py`, expected JSON schema in `output/uci2classes`, or quick-test commands for CI), tell me which section to expand and I will iterate.
