import chess

from src.puct.puct_batch import PUCTBatch
from src.puct.puct_single import PUCTSingle
from src.puct.gamestate_chess import ChessGameState
from src.puct.policyvaluenet_random import RandomPolicyValueNet


def test_puct_batch_search_returns_legal_move():
    """PUCTBatch.search should return a legal chess.Move for the given state.

    This is a lightweight smoke test: we use the random policy/value net and a
    very small number of simulations to ensure the API runs and the returned
    action is one of the legal moves for the root position.
    """
    root_state = ChessGameState()  # starting position
    policy_net = RandomPolicyValueNet()
    puct = PUCTBatch(policy_value_fn=policy_net, c_puct=1.4, batch_size=8)

    action = puct.search(root_state, num_simulations=16)

    # action must be a chess.Move and must be among legal moves of the root
    assert isinstance(action, chess.Move)
    legal = root_state.get_legal_actions()
    assert action in legal


def test_random_policyvalue_predict_batch_fallback():
    """predict_batch should return per-state priors and values matching legal move counts."""
    s1 = ChessGameState()
    s2 = ChessGameState()
    policy = RandomPolicyValueNet()

    priors_list, values_list = policy.predict_batch([s1, s2])

    assert len(priors_list) == 2
    assert len(values_list) == 2
    for s, priors in zip((s1, s2), priors_list):
        assert len(priors) == len(s.get_legal_actions())


def test_puct_single_search_returns_legal_move():
    """PUCTSingle.search should return a legal chess.Move for the given state.

    This is a lightweight smoke test: we use the random policy/value net and a
    very small number of simulations to ensure the API runs and the returned
    action is one of the legal moves for the root position.
    """
    root_state = ChessGameState()  # starting position
    policy_net = RandomPolicyValueNet()
    puct = PUCTSingle(policy_value_fn=policy_net, c_puct=1.4)

    action = puct.search(root_state, num_simulations=16)

    # action must be a chess.Move and must be among legal moves of the root
    assert isinstance(action, chess.Move)
    legal = root_state.get_legal_actions()
    assert action in legal
