# stdlib imports
import os

# pip imports
import chess

# from tmp.puct2.puct import PUCT
from src.libs.chess_model import ChessModelParams
from tmp.puct2.puct_batch import PUCTBatch
from tmp.puct2.puct_single import PUCTSingle
from tmp.puct2.gamestate_chess import ChessGameState
from tmp.puct2.policyvaluenet_mine import PolicyValueNetMine
from src.libs.chess_extra import ChessExtra
from src.utils.model_utils import ModelUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_folder_path = os.path.join(__dirname__, "../..", "output")
model_folder_path = os.path.join(output_folder_path, "model")


def play_game(num_simulations=200, max_moves=512, c_puct=1.4):
    """
    Play a full game using PUCT and a policy-value network.

    This uses the provided `ChessGameState`, `PUCT` and a dummy
    `RandomPolicyValueNet`. It loops until the chess.Board reports
    game over or `max_moves` is reached.
    """
    # 1. Create initial state
    state = ChessGameState(chess.Board())

    # 2. Create dummy NN (replace with your PyTorch model later)
    model_name = ModelUtils.MODEL_NAME.CHESS_MODEL_CONV2D
    model_params = ChessModelParams()
    model = ModelUtils.load_model(model_name, model_folder_path, model_params=model_params)
    policyValueNet = PolicyValueNetMine(model)

    # 3. Create PUCT instance
    puct = PUCTBatch(policy_value_fn=policyValueNet, c_puct=c_puct, batch_size=64)
    # puct = PUCTSingle(policy_value_fn=policyValueNet, c_puct=c_puct)
    move_num = 0
    while True:
        board = state.board
        if board.is_game_over():
            break

        if move_num >= max_moves:
            print(f"Reached max moves ({max_moves}), stopping.")
            break
        import time

        time_start = time.perf_counter()
        best_move = puct.search(state, num_simulations=num_simulations)
        time_elapsed = time.perf_counter() - time_start

        # log the performance
        print(f"Time taken for search: {time_elapsed:.2f} seconds. {num_simulations/time_elapsed:.2f} simulations/sec.")

        if best_move is None:
            print("Search returned None — stopping.")
            break

        # Normalize to a chess.Move when possible
        move_obj = best_move

        uci = move_obj.uci()
        print(f"{move_num+1}. ({uci})")
        print(f"{ChessExtra.board_to_string(board)}")  # type: ignore

        state = state.apply_action(move_obj)
        move_num += 1

    print("\nFinal board:")
    print(state.board)

    print("Result:", state.board.result())
    if state.board.is_checkmate():
        # board.turn is the side to move; the side that delivered mate is the opposite
        winner = "White" if state.board.turn == chess.BLACK else "Black"
        print("Checkmate — winner:", winner)

    return state


if __name__ == "__main__":
    play_game(num_simulations=3000)
