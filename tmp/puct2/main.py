import chess
from tmp.puct2.puct import PUCT
from tmp.puct2.gamestate_chess import ChessGameState
from tmp.puct2.policyvaluenet_mine import PolicyValueNetMine
from src.libs.chess_extra import ChessExtra


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
    model = PolicyValueNetMine()

    # 3. Create PUCT instance
    puct = PUCT(policy_value_fn=model, c_puct=c_puct)

    move_num = 0
    while True:
        board = state.board
        if board.is_game_over():
            break

        if move_num >= max_moves:
            print(f"Reached max moves ({max_moves}), stopping.")
            break

        best_move = puct.search(state, num_simulations=num_simulations)
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
    play_game(num_simulations=50)
