help: ## show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

clean: clean_output clean_pgn_tensors ## clean all generated files

clean_output: ## clean output directory
	rm -f ./output/*

clean_pgn_tensors: ## clean pgn tensors
	rm -f ./data/pgn_tensors/*

check_dataset: ## check dataset integrity
	./bin/check_dataset.py

lint: ## Run lint checker on source files
	pyright bin/**/*.py src/**/*.py

full_pipeline: clean build_dataset train play ## run the full pipeline

.PHONY: help clean build_dataset train play full_pipeline


build_dataset: ## build the dataset
	./bin/build_dataset.py

build_evals: ## build the evals for the dataset
	./bin/build_evals.py

train: ## train the model
	./bin/train.py

play: ## play against the model
	./bin/play.py
