# stdlib imports
import os
import time

# local imports
from .libs.utils import Utils
from .libs.io_utils import IOUtils

# setup __dirname__
__dirname__ = os.path.dirname(os.path.abspath(__file__))

class DatasetBuilderCommand:
    ###############################################################################
    # Create Dataset
    ###############################################################################
    @staticmethod
    def build_dataset(max_files_count: int = 15, max_games_count: int = 7000):
        output_folder_path = f"{__dirname__}/../output/"
        dataset_creation_start_time = time.time()

        # Create dataset
        boards_tensor, moves_tensor, uci_to_classindex = Utils.create_dataset(
            max_files_count=max_files_count, max_games_count=max_games_count
        )

        # Save the dataset for later
        IOUtils.save_dataset(boards_tensor, moves_tensor, uci_to_classindex, folder_path=output_folder_path)

        # display elapsed time
        dataset_creation_elapsed_time = time.time() - dataset_creation_start_time
        print(f"Dataset creation/loading time: {dataset_creation_elapsed_time:.2f} seconds")

        # Dataset creation stats
        print(Utils.dataset_summary(boards_tensor, moves_tensor, uci_to_classindex))
