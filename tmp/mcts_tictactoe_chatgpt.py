# from https://chatgpt.com/c/68d5bb3d-5fa0-8326-bc9b-f9f43a40b3c9

# Monte Carlo Tree Search (MCTS) for Tic-Tac-Toe
# Self-contained implementation:
# - TicTacToe state class
# - MCTSNode class with UCT selection
# - mcts(rootstate, itermax) returns best move
# - demo functions to play/evaluate

import random, math, copy
from typing import List

class TicTacToe:
    def __init__(self, board=None, player=1):
        # board: list of 9: 0 empty, 1 player1 (X), 2 player2 (O)
        self.board = board[:] if board is not None else [0]*9
        self.player_to_move = player  # 1 or 2

    def clone(self):
        return TicTacToe(self.board, self.player_to_move)

    def get_moves(self) -> List[int]:
        # returns list of available move indices 0..8
        return [i for i,v in enumerate(self.board) if v==0]

    def do_move(self, move:int):
        # apply move for player_to_move and switch player
        if self.board[move] != 0:
            raise ValueError("Invalid move")
        self.board[move] = self.player_to_move
        self.player_to_move = 1 if self.player_to_move==2 else 2

    def is_terminal(self):
        return self.check_winner() is not None or all(v!=0 for v in self.board)

    def check_winner(self):
        lines = [(0,1,2),(3,4,5),(6,7,8),
                 (0,3,6),(1,4,7),(2,5,8),
                 (0,4,8),(2,4,6)]
        for (a,b,c) in lines:
            if self.board[a] and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a] # 1 or 2
        if all(v!=0 for v in self.board):
            return 0  # draw
        return None # not finished

    def get_result(self, player:int) -> float:
        # returns result from the viewpoint of `player`:
        # 1.0 win for player, 0.0 loss, 0.5 draw
        winner = self.check_winner()
        if winner is None:
            return None
        if winner == 0:
            return 0.5
        return 1.0 if winner == player else 0.0

    def pretty(self):
        symbols = {0:'.',1:'X',2:'O'}
        rows = []
        for i in range(0,9,3):
            rows.append(' '.join(symbols[self.board[i+j]] for j in range(3)))
        return "\n".join(rows)


class MCTSNode:
    def __init__(self, state:TicTacToe, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move              # move that led to this node (from parent)
        self.wins = 0.0               # total reward (from viewpoint of player who JUST moved)
        self.visits = 0
        self.children = []            # list of child nodes
        self.untried_moves = state.get_moves()[:]  # moves not yet expanded
        self.player_just_moved = 1 if state.player_to_move==2 else 2

    def uct_select_child(self, c_param=1.4142135623730951):
        # select child with highest UCT value
        choices_weights = [
            (child.wins/child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def add_child(self, move, state):
        child = MCTSNode(state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        # `result` is from viewpoint of player_just_moved (1.0 win, 0.0 loss, 0.5 draw)
        self.visits += 1
        self.wins += result


def mcts(rootstate:TicTacToe, itermax=1000, verbose=False):
    rootnode = MCTSNode(rootstate)
    for i in range(itermax):
        node = rootnode
        state = rootstate.clone()

        # Selection
        while node.untried_moves == [] and node.children:
            node = node.uct_select_child()
            state.do_move(node.move)

        # Expansion
        if node.untried_moves:
            m = random.choice(node.untried_moves)
            state.do_move(m)
            node = node.add_child(m, state.clone())

        # Simulation (playout)
        while not state.is_terminal():
            state.do_move(random.choice(state.get_moves()))

        # Backpropagation
        winner = state.check_winner()  # 1/2/0
        while node is not None:
            if winner == 0:
                result = 0.5
            elif winner == node.player_just_moved:
                result = 1.0
            else:
                result = 0.0
            node.update(result)
            node = node.parent

    # choose the move with highest visits
    best_child = sorted(rootnode.children, key=lambda c: c.visits)[-1]
    if verbose:
        for c in sorted(rootnode.children, key=lambda c: c.visits, reverse=True):
            print(f"Move {c.move} | visits {c.visits} | wins {c.wins:.1f} | winrate {c.wins/c.visits:.3f}")
    return best_child.move


# Demo: MCTS vs Random
def play_game(mcts_iters=1000, verbose=True):
    state = TicTacToe()
    while not state.is_terminal():
        if state.player_to_move == 1:
            move = mcts(state, itermax=mcts_iters)
        else:
            move = random.choice(state.get_moves())
        state.do_move(move)
        if verbose:
            print(f"Player {1 if state.player_to_move==2 else 2} played {move}")
            print(state.pretty())
            print("-"*10)
    winner = state.check_winner()
    if verbose:
        if winner == 0:
            print("Result: Draw")
        else:
            print(f"Result: Player {winner} wins")
    return winner

def evaluate(n_games=10, mcts_iters=500):
    results = {1:0, 2:0, 0:0}
    for i in range(n_games):
        winner = play_game(mcts_iters=mcts_iters, verbose=False)
        results[winner] += 1
    print(f"After {n_games} games (MCTS iters={mcts_iters}): {results}")
    return results

if __name__ == "__main__":
    random.seed(1)
    print("Single demo game (MCTS vs Random):\n")
    play_game(mcts_iters=2000, verbose=True)
    print("\nRunning 20 games (quiet) to evaluate MCTS strength vs Random:\n")
    evaluate(20, mcts_iters=200)
