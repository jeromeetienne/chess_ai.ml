help: ## show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

clean_output: ## clean output directory
	rm -f ./output/*

clean_pgn_tensors: ## clean pgn tensors
	rm -f ./data/pgn_tensors/*

clean_pgn_splits: ## clean split pgn files
	rm -f ./data/pgn_splits/*.split_*.pgn

clean: clean_output clean_pgn_tensors ## clean all generated files

pgn_split_lichess:
	./scripts/pgn_splitter.py  -d ./data/pgn_splits -v ./data/pgn/lichess_elite/*.pgn

pgn_split_fishtest:
	./scripts/pgn_splitter.py  -d ./data/pgn_splits -v ./data/pgn/fishtest_stockfish/*/*/*.pgn.gz


lint: ## Run lint checker on source files
	pyright bin/**/*.py src/**/*.py

full_pipeline: clean build_dataset train play ## run the full pipeline

.PHONY: help clean build_dataset train play full_pipeline

check_dataset: ## check dataset integrity
	./scripts/check_dataset.py -fc 2

build_dataset: ## build the dataset
	./bin/build_dataset.py

build_evals_fishtest: ## build the evals for the dataset using fishtest pgns
	./scripts/build_evals_fishtest.py

build_evals_stockfish: ## build the evals for the dataset using stockfish evaluations computed live (slow)
	./scripts/build_evals_stockfish.py

train: ## train the model
	./bin/train.py

play: ## play against the model
	./bin/play.py
