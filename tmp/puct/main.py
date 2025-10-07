import chess
import chess.engine
from typing import Any, Dict, List
import random
from tmp.puct.puct import PUCT

# Reuse the Node and PUCT classes from before

class ChessGame:
    def get_legal_moves(self, state: chess.Board) -> List[chess.Move]:
        return list(state.legal_moves)

    def is_terminal(self, state: chess.Board) -> bool:
        return state.is_game_over()

    def next_state(self, state: chess.Board, move: chess.Move) -> chess.Board:
        new_board = state.copy()
        new_board.push(move)
        return new_board

    def current_player(self, state: chess.Board) -> int:
        # 1 for White, -1 for Black
        return 1 if state.turn == chess.WHITE else -1

    def evaluate_state(self, state: chess.Board) -> float:
        """
        Simple static evaluation for demonstration:
        +1 for White advantage, -1 for Black advantage
        """
        if state.is_checkmate():
            return -1.0 if state.turn == chess.WHITE else 1.0
        if state.is_stalemate() or state.is_insufficient_material():
            return 0.0

        # Simple material evaluation
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        white_score = sum(values[p.piece_type] for p in state.piece_map().values() if p.color == chess.WHITE)
        black_score = sum(values[p.piece_type] for p in state.piece_map().values() if p.color == chess.BLACK)
        return (white_score - black_score) / 39  # normalize between -1 and 1


# --- Usage ---
if __name__ == "__main__":
    board = chess.Board()
    game = ChessGame()
    puct = PUCT(game=game, n_simulations=200, c_puct=1.5)

    while not board.is_game_over():
        root = puct.search(board)
        move = puct.best_move(root)
        print(f"Move chosen: {move}")
        board.push(move)
        print(board, "\n")
        
        # For demonstration, we play random opponent move
        if not board.is_game_over():
            opponent_move = random.choice(list(board.legal_moves))
            board.push(opponent_move)
            print(f"Opponent played: {opponent_move}")
            print(board, "\n")

    print("Game over:", board.result())
