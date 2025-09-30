help: ## show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

clean: ## clean output directory
	rm -f ./output/*

lint: ## Run lint checker on source files
	pyright bin/**/*.py src/**/*.py

full_pipeline: clean build_dataset train predict play ## run the full pipeline

.PHONY: help clean build_dataset train predict play full_pipeline

build_dataset: ## build the dataset
	./bin/build_dataset.py

build_evals: ## build the evals for the dataset
	./bin/build_evals.py

train: ## train the model
	./bin/train.py

predict: ## predict a move
	./bin/predict.py

play: ## play against the model
	./bin/play.py
