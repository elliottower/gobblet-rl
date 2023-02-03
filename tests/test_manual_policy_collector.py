# adapted from https://github.com/Farama-Foundation/PettingZoo/blob/master/tutorials/Tianshou/3_cli_and_logging.py
"""
This is a full example of using Tianshou with MARL to train agents, complete with argument parsing (CLI) and logging.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""

from typing import Optional, Tuple

import gym
import numpy as np
import torch
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, MultiAgentPolicyManager


from gobblet import gobblet_v1
from gobblet.game.collector_manual_policy import ManualPolicyCollector
from gobblet.game.greedy_policy import GreedyPolicy
import time


def get_agents() -> Tuple[BasePolicy, list]:
    env = get_env()
    agents = [GreedyPolicy(), GreedyPolicy()]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, env.agents

def get_env(render_mode=None, args=None):
    return PettingZooEnv(gobblet_v1.env(render_mode=render_mode, args=args))

# ======== allows the user to input moves and play vs a pre-trained agent ======
def test_collector() -> None:
    env = DummyVectorEnv([lambda: get_env(render_mode="human", args=None)])
    policy, agents = get_agents()
    collector = ManualPolicyCollector(policy, env, exploration_noise=True) # Collector for CPU actions
    pettingzoo_env = env.workers[0].env.env

    output0 = np.array([[True, True, True, True, True, True, True, True, True, True, True, True,
                         True, True, True, True, True, True, True, True, True, True, True, True,
                         True, True, True, True, True, True, True, True, True, True, True, True,
                         True, True, True, True, True, True, True, True, True, True, True, True,
                         True, True, True, True, True, True]])
    assert(np.array_equal(collector.data.obs.mask, output0))

    ''' PLAYER 1'''
    action = np.array(18)
    result = collector.collect_result(action=action.reshape(1), render=0.1)
    output1 = np.array([[False, True, True, True, True, True, True, True, True, False, True, True,
                         True, True, True, True, True, True, False, True, True, True, True, True,
                         True, True, True, False, True, True, True, True, True, True, True, True,
                         True, True, True, True, True, True, True, True, True, True, True, True,
                         True, True, True, True, True, True]])
    assert(np.array_equal(collector.data.obs.mask, output1))

    time.sleep(.25)

    ''' PLAYER 2 (covers it)'''
    action = np.array(36)
    result = collector.collect_result(action=action.reshape(1), render=0.1)

    output2 = np.array([[False,  True,  True,  True,  True,  True,  True,  True,  True,
                        False,  True,  True,  True,  True,  True,  True,  True,  True,
                        False, False, False, False, False, False, False, False, False,
                        False,  True,  True,  True,  True,  True,  True,  True,  True,
                        False,  True,  True,  True,  True,  True,  True,  True,  True,
                        False,  True,  True,  True,  True,  True,  True,  True,  True]])
    assert np.array_equal(collector.data.obs.mask, output2)

    time.sleep(.25)

    ''' PLAYER 1'''
    action = np.array(27+1)
    result = collector.collect_result(action=action.reshape(1), render=0.1)

    output3 = np.array([[False, False,  True,  True,  True,  True,  True,  True,  True,
                        False, False,  True,  True,  True,  True,  True,  True,  True,
                        False, False,  True,  True,  True,  True,  True,  True,  True,
                        False, False,  True,  True,  True,  True,  True,  True,  True,
                        False,  True,  True,  True,  True,  True,  True,  True,  True,
                        False,  True,  True,  True,  True,  True,  True,  True,  True]])
    assert(np.array_equal(collector.data.obs.mask, output3))

    time.sleep(.25)
    ''' PLAYER 2 (covers it)'''
    action = np.array(45+1)
    result = collector.collect_result(action=action.reshape(1), render=0.1)

    output4 = np.array([[False, False,  True,  True,  True,  True,  True,  True,  True,
                        False, False,  True,  True,  True,  True,  True,  True,  True,
                        False, False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False, False,
                        False, False,  True,  True,  True,  True,  True,  True,  True,
                        False, False,  True,  True,  True,  True,  True,  True,  True]])
    assert(np.array_equal(collector.data.obs.mask, output4))

    time.sleep(.25)

    ''' PLAYER 1 (tries to move covered piece [ILLEGAL])'''
    action = np.array(27+2)
    # Moves 18-35 should be illegal as they are with medium pieces. (36 and 37 as well but they are with a large piece)

    output6 = [2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 47, 48, 49, 50, 51, 52, 53]
    legal_moves = pettingzoo_env.unwrapped._legal_moves()

    assert(output6 == legal_moves)

    output5 = np.array([[False, False,  True,  True,  True,  True,  True,  True,  True,
                        False, False,  True,  True,  True,  True,  True,  True,  True,
                        False, False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False, False,
                        False, False,  True,  True,  True,  True,  True,  True,  True,
                        False, False,  True,  True,  True,  True,  True,  True,  True]])
    assert(np.array_equal(collector.data.obs.mask, output5))

    result = collector.collect_result(action=action.reshape(1), render=0.1)
    # Result should be empty because this is an illegal move

    output7 = {'n/ep': 0, 'n/st': 1, 'rews': np.array([], dtype=np.float64), 'lens': np.array([], dtype=np.int64), 'idxs': np.array([], dtype=np.int64), 'rew': 0, 'len': 0, 'rew_std': 0, 'len_std': 0}
    assert(str(result) == str(output7))

    # Board state should be unchanged because the bot tried to execute an illegal move"
    output8 = np.array([[[ 0.,  0.,  0.],
                        [ 0.,  0.,  0.],
                        [ 0.,  0.,  0.]],
                       [[ 3.,  4.,  0.],
                        [ 0.,  0.,  0.],
                        [ 0.,  0.,  0.]],
                       [[-5., -6.,  0.],
                        [ 0.,  0.,  0.],
                        [ 0.,  0.,  0.]]])
    assert(np.array_equal(pettingzoo_env.unwrapped.board.squares.reshape(3,3,3), output8))


if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    print("Starting game...")
    test_collector()