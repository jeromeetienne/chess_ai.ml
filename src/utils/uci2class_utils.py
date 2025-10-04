import chess
import os
import json
import typing

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.join(__dirname__, "../../output")
uci2classes_folder_path = os.path.join(output_folder_path, "uci2classes")

class Uci2ClassUtils:
    """
    Utility class to handle UCI to class index mappings and vice versa.
    """

    _uci2class_white: dict[str, int] | None = None
    _uci2class_black: dict[str, int] | None = None
    _class2uci_white: dict[int, str] | None = None
    _class2uci_black: dict[int, str] | None = None

    @staticmethod
    def _init_if_needed():
        """
        Load the UCI to class mappings from JSON files if they haven't been loaded yet.
        """
        is_initialized = Uci2ClassUtils._uci2class_white is not None
        if is_initialized:
            return

        # load the file for white
        white_file_path = os.path.join(uci2classes_folder_path, f"uci2class_arr_white.json")
        with open(white_file_path, "r") as file_reader:
            uci2class_arr: list[str] = json.load(file_reader)
        # Build the mapping for white
        Uci2ClassUtils._uci2class_white = {move_uci: index for index, move_uci in enumerate(uci2class_arr)}
        # build the inverse mapping for white
        Uci2ClassUtils._class2uci_white = {index: move_uci for index, move_uci in enumerate(uci2class_arr)}

        # load the file for black
        black_file_path = os.path.join(uci2classes_folder_path, f"uci2class_arr_black.json")
        with open(black_file_path, "r") as file_reader:
            uci2class_arr: list[str] = json.load(file_reader)
        # Build the mapping for black
        Uci2ClassUtils._uci2class_black = {move_uci: index for index, move_uci in enumerate(uci2class_arr)}
        # build the inverse mapping for black
        Uci2ClassUtils._class2uci_black = {index: move_uci for index, move_uci in enumerate(uci2class_arr)}

    @staticmethod
    def get_uci2class(color: chess.Color) -> dict[str, int]:
        # Initialize the mappings if needed
        Uci2ClassUtils._init_if_needed()

        # get the appropriate mapping based on the color
        if color == chess.WHITE:
            uci2class = typing.cast(dict[str, int], Uci2ClassUtils._uci2class_white)
        else:
            uci2class = typing.cast(dict[str, int], Uci2ClassUtils._uci2class_black)
        # return the mapping
        return uci2class
    
    @staticmethod
    def get_class2uci(color: chess.Color) -> dict[int, str]:
        # Initialize the mappings if needed
        Uci2ClassUtils._init_if_needed()
        
        # get the appropriate mapping based on the color
        if color == chess.WHITE:
            class2uci = typing.cast(dict[int, str], Uci2ClassUtils._class2uci_white)
        else:
            class2uci = typing.cast(dict[int, str], Uci2ClassUtils._class2uci_black)
        # return the mapping
        return class2uci
