help: ## show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

lint: ## Run lint checker on source files
	pyright bin/**/*.py src/**/*.py

full_pipeline: clean build_boards_moves build_evals_outcome train play ## run the full pipeline

train_deep_20: ## train a model with first 20 pgn files
	./bin/build_boards_moves.py -fc 20
	./bin/build_evals_fishtest.py -fc 20
	./bin/train.py -fc 20

train_deep_25: ## train a model with first 25 pgn files
	./bin/build_boards_moves.py -fc 25
	./bin/build_evals_fishtest.py -fc 25
	./bin/train.py -fc 25

train_deep_35: ## train a model with first 35 pgn files
	./bin/build_boards_moves.py -fc 35
	./bin/build_evals_fishtest.py -fc 35
	./bin/train.py -fc 35

bench_inference: ## benchmark the model
	./scripts/bench_inference.py

check_dataset: ## check dataset integrity
	./scripts/check_dataset.py -fc 2

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
	./scripts/pgn_splitter.py  -d ./output/pgn_splits -v ./data/lichess_elite/*.pgn

pgn_split_fishtest:
	./scripts/pgn_splitter.py  -d ./output/pgn_splits -v ./data/fishtest_stockfish/*/*/*.pgn.gz


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

###############################################################################
#   Model targets
#
train: ## train the model
	./bin/train.py

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
	./scripts/ucinet_named_pipe_proxy.py
