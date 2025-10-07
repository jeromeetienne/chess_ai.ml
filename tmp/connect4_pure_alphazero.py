"""
AlphaZero-style PUCT MCTS for Connect Four
Single-file minimal but functional implementation suitable for experimentation.

Features:
- Connect4 environment (6x7)
- Small PyTorch residual-like network that outputs policy logits and a scalar value
- PUCT-based MCTS using network priors and value estimates
- Self-play loop that collects (state, pi, z) training examples
- Simple replay buffer and training step

Dependencies:
- Python 3.8+
- numpy
- torch

Usage:
- Run the file `python connect4_puct_alpha_zero.py`
- It will run a tiny self-play + train loop by default (very small for speed).
- Tune hyperparameters at the bottom of the file.

This file is intentionally compact and educational rather than optimized for performance.
"""

import math
import random
from collections import defaultdict, deque, namedtuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# ------------------ Connect4 Environment ------------------
class Connect4:
    ROWS = 6
    COLS = 7

    def __init__(self):
        # board: 0 empty, 1 player1, -1 player2
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = 1
        self.last_move = None
        self.move_count = 0

    def clone(self):
        c = Connect4()
        c.board = self.board.copy()
        c.current_player = self.current_player
        c.last_move = self.last_move
        c.move_count = self.move_count
        return c

    def legal_moves(self):
        # return list of columns that are not full
        return [c for c in range(self.COLS) if self.board[0, c] == 0]

    def play(self, col):
        if self.board[0, col] != 0:
            raise ValueError('Column is full')
        for r in range(self.ROWS - 1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = self.current_player
                self.last_move = (r, col)
                self.current_player *= -1
                self.move_count += 1
                return

    def is_full(self):
        return self.move_count >= self.ROWS * self.COLS

    def check_winner(self):
        # returns 1 if player1 wins, -1 if player2 wins, 0 otherwise
        b = self.board
        R, C = self.ROWS, self.COLS
        for r in range(R):
            for c in range(C):
                if b[r, c] == 0:
                    continue
                p = b[r, c]
                # right
                if c + 3 < C and all(b[r, c + k] == p for k in range(4)):
                    return p
                # down
                if r + 3 < R and all(b[r + k, c] == p for k in range(4)):
                    return p
                # down-right
                if r + 3 < R and c + 3 < C and all(b[r + k, c + k] == p for k in range(4)):
                    return p
                # down-left
                if r + 3 < R and c - 3 >= 0 and all(b[r + k, c - k] == p for k in range(4)):
                    return p
        return 0

    def result(self):
        w = self.check_winner()
        if w != 0:
            return w
        if self.is_full():
            return 0
        return None

    def canonical_board(self):
        # return a 2x6x7 representation: channel0 = current player pieces, channel1 = opponent pieces
        cur = (self.board == self.current_player).astype(np.float32)
        opp = (self.board == -self.current_player).astype(np.float32)
        return np.stack([cur, opp], axis=0)

    def __str__(self):
        s = ''
        for r in range(self.ROWS):
            s += ' '.join('.XO'[1 if self.board[r,c]==1 else 2 if self.board[r,c]==-1 else 0] for c in range(self.COLS)) + '\n'
        return s

# ------------------ Neural Network ------------------
class SmallResNet(nn.Module):
    def __init__(self, in_channels=2, channels=64, board_h=6, board_w=7, n_actions=7):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        # a couple of residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
            )
            for _ in range(3)
        ])

        self.relu = nn.ReLU()

        # policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_h * board_w, n_actions)

        # value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_h * board_w, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: batch x 2 x 6 x 7
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            residual = out
            out = block(out)
            out = self.relu(out + residual)

        # policy
        p = self.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # value
        v = self.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = self.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return p, v

# ------------------ MCTS Node ------------------
class MCTSNode:
    def __init__(self, prior=0.0):
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.children = {}  # action -> node
        self.is_expanded = False

    def Q(self):
        return 0.0 if self.N == 0 else self.W / self.N

# ------------------ PUCT MCTS ------------------
class PUCTMCTS:
    def __init__(self, net, c_puct=1.5, n_simulations=160, dirichlet_alpha=0.3, eps=0.25, device='cpu'):
        self.net = net
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dir_alpha = dirichlet_alpha
        self.eps = eps
        self.device = device

    def run(self, env: Connect4):
        root = MCTSNode()
        # expand root
        state = env.canonical_board()
        legal = env.legal_moves()
        priors, _ = self._net_eval(state)
        priors = priors.cpu().numpy()

        # mask illegal moves
        mask = np.zeros_like(priors)
        mask[legal] = 1
        priors = priors * mask
        if priors.sum() == 0:
            # when net gives zero to legal moves (rare), assign uniform
            priors[legal] = 1.0
        priors = priors / priors.sum()

        # add dirichlet noise to root priors
        noise = np.random.dirichlet([self.dir_alpha] * len(priors))
        priors = (1 - self.eps) * priors + self.eps * noise

        for a, p in enumerate(priors):
            if mask[a]:
                root.children[a] = MCTSNode(prior=float(p))

        # run simulations
        for _ in range(self.n_simulations):
            self._simulate(env, root)

        # return visit counts as policy target
        visit_counts = np.array([root.children[a].N if a in root.children else 0 for a in range(env.COLS)], dtype=np.float32)
        if visit_counts.sum() > 0:
            pi = visit_counts / visit_counts.sum()
        else:
            # fallback uniform
            legal = env.legal_moves()
            pi = np.zeros(env.COLS, dtype=np.float32)
            pi[legal] = 1.0 / len(legal)
        return pi, root

    def _simulate(self, env: Connect4, root: MCTSNode):
        path = []  # list of (node, action, env_state)
        node = root
        sim_env = env.clone()

        # selection
        while True:
            if not node.is_expanded:
                break
            # pick best action
            total_N = sum(child.N for child in node.children.values())
            best_score = -float('inf')
            best_action = None
            for a, child in node.children.items():
                U = child.Q() + self.c_puct * child.P * math.sqrt(total_N) / (1 + child.N)
                if U > best_score:
                    best_score = U
                    best_action = a
            if best_action is None:
                break
            path.append((node, best_action))
            sim_env.play(best_action)
            # check terminal
            r = sim_env.result()
            node = node.children.get(best_action)
            if node is None:
                # action hasn't been expanded in tree yet
                node = MCTSNode()
                break
            if r is not None:
                break

        # expansion/evaluation
        r = sim_env.result()
        if r is not None:
            # terminal
            if r == 0:
                value = 0.0
            else:
                # value from perspective of current player in sim_env (after move)
                value = 1.0 if r == sim_env.current_player * -1 else -1.0
        else:
            state = sim_env.canonical_board()
            priors, v = self._net_eval(state)
            priors = priors.cpu().numpy()
            legal = sim_env.legal_moves()
            mask = np.zeros_like(priors)
            mask[legal] = 1
            priors = priors * mask
            if priors.sum() == 0:
                priors[legal] = 1.0
            priors = priors / priors.sum()

            # create children
            node.is_expanded = True
            for a in legal:
                node.children[a] = MCTSNode(prior=float(priors[a]))
            value = float(v.cpu().numpy())
            # value is from perspective of the current player in sim_env

        # backpropagate
        # value is for the player to move at sim_env (after path actions). We need to backpropagate
        # to the root, flipping sign each move because players alternate.
        for parent, action in reversed(path):
            parent.N += 1
            # if flipping: when backing up one ply, the value is from opponent perspective
            parent.W += value
            value = -value
        # also update the final node along the last action if exists in path
        if path:
            last_parent, last_action = path[-1]
            child = last_parent.children[last_action]
            child.N += 1
            child.W += value if False else (-value)  # already flipped above; but to keep logic consistent, we'll add -value
            # Note: this line ensures the child accumulates the visit. It's somewhat redundant with parent's update but keeps counts coherent.

    def _net_eval(self, state_array):
        # state_array: 2x6x7 numpy
        x = torch.from_numpy(state_array).unsqueeze(0).to(self.device)  # 1x2x6x7
        self.net.eval()
        with torch.no_grad():
            logits, v = self.net(x)
            probs = F.softmax(logits, dim=-1).squeeze(0)
        return probs, v.detach()

# ------------------ Replay Buffer ------------------
Transition = namedtuple('Transition', ['state', 'pi', 'z'])

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.stack([t.state for t in batch])
        pis = np.stack([t.pi for t in batch])
        zs = np.array([t.z for t in batch], dtype=np.float32)
        return states, pis, zs

    def __len__(self):
        return len(self.buffer)

# ------------------ Trainer ------------------
class AlphaZeroTrainer:
    def __init__(self, net, device='cpu'):
        self.device = device
        self.net = net.to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.mcts = PUCTMCTS(self.net, c_puct=1.5, n_simulations=100, device=device)
        self.replay = ReplayBuffer(5000)

    def self_play_game(self, temp=1.0):
        env = Connect4()
        states = []
        pis = []
        players = []

        while True:
            pi, root = self.mcts.run(env)
            # temperature sampling for exploration early in the game
            if temp == 0:
                action = int(np.argmax(pi))
            else:
                probs = pi ** (1.0 / temp)
                probs = probs / probs.sum()
                action = int(np.random.choice(len(probs), p=probs))

            states.append(env.canonical_board())
            pis.append(pi)
            players.append(env.current_player)

            env.play(action)
            r = env.result()
            if r is not None:
                # compute z for each state (from perspective of player who moved at that state)
                for s, p, pl in zip(states, pis, players):
                    if r == 0:
                        z = 0.0
                    else:
                        z = 1.0 if r == pl else -1.0
                    self.replay.push(s, p, z)
                return r

    def train_step(self, batch_size=32, epochs=1):
        if len(self.replay) < batch_size:
            return None
        self.net.train()
        states, pis, zs = self.replay.sample(batch_size)
        states = torch.from_numpy(states).to(self.device)
        pis = torch.from_numpy(pis).to(self.device)
        zs = torch.from_numpy(zs).to(self.device)

        logits, values = self.net(states)
        loss_policy = -torch.mean(torch.sum(pis * F.log_softmax(logits, dim=-1), dim=1))
        loss_value = F.mse_loss(values, zs)
        loss = loss_policy + loss_value

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optimizer.step()
        return loss.item()

# ------------------ Simple CLI / Run Loop ------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = SmallResNet()
    trainer = AlphaZeroTrainer(net, device=device)

    n_iterations = 20
    self_play_per_iter = 10
    train_steps_per_iter = 50

    for it in range(n_iterations):
        print(f"Iteration {it+1}/{n_iterations}")
        # self-play
        results = {1:0, -1:0, 0:0}
        for _ in range(self_play_per_iter):
            r = trainer.self_play_game(temp=1.0 if random.random() < 0.9 else 0)
            results[r] += 1
        print(" Self-play results:", results, "replay size:", len(trainer.replay))

        # train
        for step in range(train_steps_per_iter):
            loss = trainer.train_step(batch_size=64)
            if loss is not None and step % 10 == 0:
                print(f"  Train step {step}	loss={loss:.4f}")

    print('Done')

if __name__ == '__main__':
    main()
