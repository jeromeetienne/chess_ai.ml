# stdlib imports
import os
import time

# local imports
from libs.utils import Utils
from libs.io_utils import IOUtils

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))

###############################################################################
# Create Dataset
#

def build_dataset():
        output_folder_path = f"{__dirname__}/../output/"
        dataset_creation_start_time = time.time()

        # Create dataset
        boards_tensor, best_move_tensor, uci_to_classindex = Utils.create_dataset()

        # Save the dataset for later
        IOUtils.save_dataset(boards_tensor, best_move_tensor, uci_to_classindex, folder_path=output_folder_path)

        # display elapsed time
        dataset_creation_elapsed_time = time.time() - dataset_creation_start_time
        print(f"Dataset creation/loading time: {dataset_creation_elapsed_time:.2f} seconds")

        # Dataset creation stats
        print(f"Total boards in dataset: {len(boards_tensor)}")
        print(f"Number of classes: {len(uci_to_classindex)}")

###############################################################################
###############################################################################
#	 Main entry point
###############################################################################
###############################################################################

if __name__ == "__main__":
    
    build_dataset()