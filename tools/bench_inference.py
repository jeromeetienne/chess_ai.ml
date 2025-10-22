#!/usr/bin/env python3

# stdlib imports
import os
import time

# pip imports
import argparse
import torch

# local imports
from src.libs.chess_model import ChessModelParams
from src.encoding.board_encoding import BoardEncoding
from src.utils.model_utils import ModelUtils

# __dirname__ = os.path.dirname(os.path.abspath(__file__))
# output_folder_path = os.path.join(__dirname__, "..", "output")
# data_folder_path = os.path.abspath(os.path.join(__dirname__, "..", "data"))
# tensors_folder_path = os.path.join(data_folder_path, "pgn_tensors")

###############################################################################
#   Main entry point
#
if __name__ == "__main__":
    # Parse command line arguments
    argParser = argparse.ArgumentParser(description="Benchmark different model architectures.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # add positional arguments model_names with choices and allow multiple values
    argParser.add_argument(
        "model_names",
        nargs="*",
        # choices=ModelUtils.get_supported_models(),
        default=ModelUtils.get_supported_models(),
        help="One or more model architecture names to benchmark. If omitted, all supported models will be benchmarked.",
    )
    args = argParser.parse_args()

    # =============================================================================
    # sanity check
    # =============================================================================
    for model_name in args.model_names:
        if model_name not in ModelUtils.get_supported_models():
            print(f'Invalid model name: {model_name}. Supported models are: {", ".join(ModelUtils.get_supported_models())}')

    ###############################################################################
    #   Load and setup model
    #

    print(f'Benchmarking models: {", ".join(args.model_names)}  ')

    # allow benchmarking multiple models in sequence
    for model_name in args.model_names:
        print(f"\nBenchmarking model: {model_name}")

        # Create the model
        model = ModelUtils.create_model(model_name, model_params=ChessModelParams())

        # =============================================================================
        # Load model weights if available
        # =============================================================================
        # set to eval mode
        model.eval()

        # Check for GPU
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore

        # Move model to the appropriate device
        model = model.to(device)

        ###############################################################################
        #   Create a single dummy input (batch size of 1)
        #

        # get an random input
        batch_size = 1
        test_input = torch.randn(batch_size, *BoardEncoding.get_input_shape(), dtype=torch.float32)
        test_input = test_input.to(device)

        ###############################################################################
        #   Model warmup
        #
        with torch.no_grad():
            outputs = model(test_input)

        ###############################################################################
        #   Benchmarking
        #
        time_start = time.perf_counter()

        inference_count = 10_000
        for inference_index in range(1, inference_count + 1):
            with torch.no_grad():
                outputs = model(test_input)
            if inference_index % 1000 == 0 or inference_index == inference_count - 1:
                time_elapsed = time.perf_counter() - time_start
                print(f"\rPrediction {inference_index:4}/{inference_count}  {(inference_index / time_elapsed):.0f} predictions/sec", end="", flush=True)
        print()  # newline after progress output
        time_elapsed = time.perf_counter() - time_start

        ###############################################################################
        #   Display results for this model
        #
        inference_per_sec = inference_count / time_elapsed
        print(f"Time taken for {inference_count} predictions: {time_elapsed:.2f} sec. {inference_per_sec:.0f} predictions/sec")
