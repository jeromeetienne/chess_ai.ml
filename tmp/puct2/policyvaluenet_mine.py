# stdliv imports
import os
import random
import typing
from typing import Any, List, Tuple, Optional

# pip imports
import torch
import numpy as np

# local imports
from .gamestate_abc import GameState
from .gamestate_chess import ChessGameState
from .policyvaluenet_abc import PolicyValueNet
from src.utils.model_utils import ModelUtils
from src.libs.encoding import Encoding
from src.utils.uci2class_utils import Uci2ClassUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.join(__dirname__, "../..", "output")
model_folder_path = os.path.join(output_folder_path, "model")


class PolicyValueNetMine(PolicyValueNet):
    def __init__(self):
        super().__init__()
        model_name = "ChessModelConv2d"  # or "ChessModelResNet"
        # Load the model
        model = ModelUtils.create_model(model_name)
        ModelUtils.load_weights(model, model_folder_path)

        self._model = model
        self._device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore

    def predict(self, state: GameState) -> Tuple[List[float], float]:
        # Set the model to evaluation mode
        self._model.eval()
        # move the model to the device
        self._model.to(self._device)

        chessState: ChessGameState = typing.cast(ChessGameState, state)
        board = chessState.board
        board_tensor = Encoding.board_to_tensor(board)

        # add the batch dimension and move to the device
        board_tensor = board_tensor.unsqueeze(0).to(self._device)

        # Disable gradient calculation for inference
        with torch.no_grad():
            output = self._model(board_tensor)
            moves_preds: torch.Tensor = output[0]  # Get the move predictions
            evals_preds: torch.Tensor = output[1]  # Get the eval predictions

        moves_preds = moves_preds.squeeze(0)  # Remove batch dimension
        evals_preds = evals_preds.squeeze(0)  # Remove batch dimension

        move_probabilities = torch.softmax(moves_preds, dim=0).cpu().numpy()  # Convert to probabilities

        # priors is the probabilities of each legal move in the order of legal_moves
        priors = []
        for legal_move in list(board.legal_moves):
            uci2class = Uci2ClassUtils.get_uci2class(board.turn)
            move_index = uci2class[legal_move.uci()]
            move_probability = move_probabilities[move_index]
            priors.append(move_probability)

        # value is in [-1, 1]
        evals_value = evals_preds.item()

        return priors, evals_value
