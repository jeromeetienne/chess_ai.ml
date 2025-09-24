help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

clean: ## clean output directory
	rm -f ./output/*

build_dataset: ## build the dataset
	python src/bin/build_dataset.py

train: ## train the model
	python src/bin/train.py

predict: ## predict a move
	python src/bin/predict.py

play: ## play against the model
	python src/bin/play.py

full_pipeline: clean build_dataset train predict play ## run the full pipeline