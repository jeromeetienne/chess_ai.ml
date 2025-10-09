import random
from .gamestate_abc import GameState
from .policyvaluenet_abc import PolicyValueNet

class RandomPolicyValueNet(PolicyValueNet):
    def predict(self, state: GameState):
        legal_moves = state.get_legal_actions()
        n = len(legal_moves)
        priors = [1.0 / n] * n  # uniform priors
        value = random.uniform(-1, 1)  # random evaluation
        return priors, value
