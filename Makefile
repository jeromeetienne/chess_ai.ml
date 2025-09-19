help:
	@echo "Available targets:"
	@echo "  help     Show this help message"
	@echo "  clear    Remove all files in ./output/"
	@echo "  train    Run the training script"
	@echo "  predict  Run the prediction script"

clear:
	rm ./output/*

train:
	python src/train.py

predict:
	python src/predict.py