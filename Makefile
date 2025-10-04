help: ## show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

lint: ## Run lint checker on source files
	pyright bin/**/*.py src/**/*.py

full_pipeline: clean build_boards_moves build_evals_fishtest train play ## run the full pipeline

bench: ## benchmark the model
	./scripts/model_bench.py

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

clean: clean_model clean_pgn_tensors ## clean all generated files

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

build_evals_fishtest: ## build the evals for the dataset using fishtest pgns
	./bin/build_evals_fishtest.py

build_evals_stockfish: ## build the evals for the dataset using stockfish evaluations computed live (slow)
	./bin/build_evals_stockfish.py

###############################################################################
#   Model targets
#
train: ## train the model
	./bin/train.py

play: ## play against the model
	./bin/play.py
