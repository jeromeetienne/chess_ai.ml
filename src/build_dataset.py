# stdlib imports
import os
import time

# pip imports
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# local imports
from libs.dataset import ChessDataset
from libs.model import ChessModel
from libs.utils import Utils
from libs.io_utils import IOUtils

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))

###############################################################################
# Load/Create Dataset
#

output_folder_path = f"{__dirname__}/../output/"

dataset_creation_start_time = time.time()

# Create dataset
boards_tensor, best_move_tensor, uci_to_classindex = Utils.create_dataset()

# Save the dataset for later
IOUtils.save_dataset(boards_tensor, best_move_tensor, uci_to_classindex, folder_path=output_folder_path)

# Dataset creation stats
print(f"Total boards in dataset: {len(boards_tensor)}")
dataset_creation_elapsed_time = time.time() - dataset_creation_start_time
print(f"Dataset creation/loading time: {dataset_creation_elapsed_time:.2f} seconds")
num_classes = len(uci_to_classindex)
print(f"Number of classes: {num_classes}")

