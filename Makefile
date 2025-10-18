help: ## show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ===============================================================================
#   Test targets
# ===============================================================================

lint: ## Run lint checker on source files
	pyright bin/ src/ tools/ tests/

pytest: ## Run pytest on test files
	pytest -v tests/

test: lint pytest ## Run all tests

# ===============================================================================
#   Misc
# ===============================================================================

test_full_pipeline: clean ## run the full pipeline
	./bin/build_boards_moves.py -fc 3
	./bin/build_evals_outcome.py -fc 3
	./bin/train.py -me 3
	./bin/play.py -mp 20

bench_inference: ## benchmark the model
	./tools/bench_inference.py

check_dataset: ## check dataset integrity
	./tools/check_dataset.py -fc 2

###############################################################################
#   Clean targets
#
clean_model: ## clean output directory
	rm -f ./output/model/*

clean_pgn_tensors: ## clean pgn tensors
	rm -f ./output/pgn_tensors/*

clean_pgn_splits: ## clean split pgn files
	rm -f ./output/pgn_splits/*.split_*.pgn

clean: clean_model clean_pgn_tensors ## clean model and pgn tensors

###############################################################################
#   PGN splitting targets
#
pgn_split_lichess:
	./tools/pgn_splitter.py  -d ./output/pgn_splits -v ./data/lichess_elite/*.pgn

pgn_split_fishtest:
	./tools/pgn_splitter.py  -d ./output/pgn_splits -v ./data/fishtest_stockfish/*/*/*.pgn.gz


###############################################################################
#   Build dataset targets
#
build_boards_moves: ## build the dataset
	./bin/build_boards_moves.py

build_evals_outcome: ## build the evals for the dataset using game outcome
	./bin/build_evals_outcome.py

build_evals_fishtest: ## build the evals for the dataset using fishtest pgns
	./bin/build_evals_fishtest.py

build_evals_stockfish: ## build the evals for the dataset using stockfish evaluations computed live (slow)
	./bin/build_evals_stockfish.py

build_dataset_outcome_10: ## build dataset with 10 files
	./bin/build_boards_moves.py -fc 10
	./bin/build_evals_outcome.py -fc 10

build_dataset_outcome_20: ## build dataset with 20 files
	./bin/build_boards_moves.py -fc 20
	./bin/build_evals_outcome.py -fc 20

build_dataset_outcome_30: ## build dataset with 30 files
	./bin/build_boards_moves.py -fc 30
	./bin/build_evals_outcome.py -fc 30

build_dataset_outcome_40: ## build dataset with 40 files
	./bin/build_boards_moves.py -fc 40
	./bin/build_evals_outcome.py -fc 40	

build_dataset_outcome_50: ## build dataset with 50 files
	./bin/build_boards_moves.py -fc 50
	./bin/build_evals_outcome.py -fc 50
###############################################################################
#   train targets
#
train: ## train the model
	./bin/train.py

train_10: build_dataset_outcome_10	## train the model with 10 files
	./bin/train.py -fc 10

train_20: build_dataset_outcome_20	## train the model with 20 files
	./bin/train.py -fc 20

train_30: build_dataset_outcome_30	## train the model with 30 files
	./bin/train.py -fc 30

train_40: build_dataset_outcome_40	## train the model with 40 files
	./bin/train.py -fc 40

train_50: build_dataset_outcome_50	## train the model with 50 files
	./bin/train.py -fc 50

###############################################################################
#   play targets
#

play: play_stockfish ## play the model vs stockfish

play_human: ## play against the model
	./bin/play.py -color black -o human

play_stockfish: ## play the model vs stockfish
	./bin/play.py

###############################################################################
#   UCI protocol targets
#
ucinet_engine: ## run the model as a uci engine
	./bin/ucinet_engine.py -m named_pipe

ucinet_named_pipe_proxy: ## run a named pipe proxy for uci engine
	./tools/ucinet_named_pipe_proxy.py
