#!/usr/bin/env python3

# stdlib imports
import os
import time

# pip imports
import argparse
import torch

# local imports
from src.libs.chess_model import ChessModelConv2d, ChessModelResNet
from src.libs.encoding import Encoding

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.join(__dirname__, "..", "output")
data_folder_path = os.path.abspath(os.path.join(__dirname__, "..", "data"))
tensors_folder_path = os.path.join(data_folder_path, "pgn_tensors")

###############################################################################
#   Main entry point
#
if __name__ == "__main__":
    # Parse command line arguments
    argParser = argparse.ArgumentParser(description="Benchmark different model architectures.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argParser.parse_args()

    ###############################################################################
    #   Load and setup model
    #

    # Create the model
    model_name = "ChessModelResNet"
    if model_name == "ChessModelConv2d":
        model = ChessModelConv2d(input_shape=Encoding.get_input_shape(), output_shape=Encoding.get_output_shape())
    elif model_name == "ChessModelResNet":
        model = ChessModelResNet(input_shape=Encoding.get_input_shape(), output_shape=Encoding.get_output_shape())
    else:
        assert False, f"Unknown model name: {model_name}"

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
    test_input = torch.randn(batch_size, *Encoding.get_input_shape(), dtype=torch.float32)
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
    for inference_index in range(inference_count):
        with torch.no_grad():
            outputs = model(test_input)
        if inference_index % 1000 == 0:
            print(f"Inference {inference_index:4}/{inference_count}")

    time_elapsed = time.perf_counter() - time_start

    ###############################################################################
    #   Display results
    #
    inference_per_sec = inference_count / time_elapsed
    print(f"Time taken for {inference_count} predictions: {time_elapsed:.4f} sec. {inference_per_sec:.2f} inferences/sec.")
