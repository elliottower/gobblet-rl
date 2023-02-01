# adapted from https://github.com/Farama-Foundation/PettingZoo/blob/master/tutorials/Tianshou/3_cli_and_logging.py
"""
This is a full example of using Tianshou with MARL to train agents, complete with argument parsing (CLI) and logging.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""

import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

from gobblet import gobblet_v1
from gobblet.game.collector_manual_policy import ManualPolicyCollector
import time


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument("--render_mode", type=str, default="human", choices=["human","rgb_array", "text", "text_full"], help="Choose the rendering mode for the game.")
    parser.add_argument("--debug", action="store_true", help="enable to print extra debugging info")
    parser.add_argument("--self_play", action="store_true", help="enable training via self-play (as opposed to fixed opponent)")
    parser.add_argument("--cpu-players", type=int, default=2, choices=[1, 2], help="Number of CPU players (options: 1, 2)")
    parser.add_argument("--player", type=int, default=0, choices=[0,1], help="Choose which player to play as: red = 0, yellow = 1")
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.6,
        help="the expected winning rate: Optimal policy can get 0.7",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, " "watch the play of pre-trained models",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=2,
        help="the learned agent plays as the"
        " agent_id-th player. Choices are 1 and 2.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="the path of agent pth file " "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file "
        "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_agents(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = (
        observation_space["observation"].shape or observation_space["observation"].n
    )
    args.action_shape = env.action_space.shape or env.action_space.n
    if agent_learn is None:
        # model
        net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        agent_learn = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq,
        )
        if args.resume_path:
            agent_learn.load_state_dict(torch.load(args.resume_path))

    if agent_opponent is None:
        if args.self_play:
            agent_opponent = deepcopy(agent_learn)
        elif args.opponent_path:
            agent_opponent = deepcopy(agent_learn)
            agent_opponent.load_state_dict(torch.load(args.opponent_path))
        else:
            agent_opponent = RandomPolicy()

    if args.agent_id == 1:
        agents = [agent_learn, agent_opponent]
    else:
        agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def get_env(render_mode=None, args=None):
    return PettingZooEnv(gobblet_v1.env(render_mode=render_mode, args=args))


def train_agent(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:
    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== agent setup =========
    policy, optim, agents = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent, optim=optim
    )

    # ======== collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, "gobblet", "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # ======== callback functions used during training =========
    def save_best_fn(policy):
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, "gobblet", "dqn", "policy.pth"
            )
        torch.save(
            policy.policies[agents[args.agent_id - 1]].state_dict(), model_save_path
        )

    def stop_fn(mean_rewards):
        return mean_rewards >= args.win_rate

    def train_fn(epoch, env_step):
        policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)

    def train_fn_selfplay(epoch, env_step):
        policy.policies[agents[:]].set_eps(args.eps_train) # Same as train_fn but for both agents instead of only learner

    def test_fn(epoch, env_step):
        policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)

    def test_fn_selfplay(epoch, env_step):
        policy.policies[agents[:]].set_eps(args.eps_test) # Same as test_fn but for both agents instead of only learner


    def reward_metric(rews):
        return rews[:, args.agent_id - 1]

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn if not args.self_play else train_fn_selfplay,
        test_fn=test_fn if not args.self_play else train_fn_selfplay,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    return result, policy.policies[agents[args.agent_id - 1]]


# ======== a test function that tests a pre-trained agent ======
def watch(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
) -> None:
    env = DummyVectorEnv([lambda: get_env(render_mode=args.render_mode, args=args)])
    policy, optim, agents = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent
    )
    policy.eval()
    if not args.self_play:
        policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
    else:
        policy.policies[agents[:]].set_eps(args.eps_test)
    collector = Collector(policy, env, exploration_noise=True)

    # First step (while loop stopping conditions are not defined until we run the first step)
    result = collector.collect(n_step=1, render=args.render)
    time.sleep(0.25)

    while not (collector.data.terminated or collector.data.truncated):
        result = collector.collect(n_step=1, render=args.render)
        time.sleep(0.25) # Slow down rendering so the actions can be seen sequentially (otherwise moves happen too fast)

    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, args.agent_id - 1].mean()}, length: {lens.mean()}")

# TODO: Look more into Tianshou and see if self play is possible
# For watching I think it could just be the same policy for both agents, but for training I think self play would be different
def watch_selfplay(args, agent):
    raise NotImplementedError()
    env = DummyVectorEnv([lambda: get_env(render_mode=args.render_mode, debug=args.debug)])
    agent.set_eps(args.eps_test)
    policy = MultiAgentPolicyManager([agent, deepcopy(agent)], env) # fixed here
    policy.eval()
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")


# ======== allows the user to input moves and play vs a pre-trained agent ======
def play(
        args: argparse.Namespace = get_args(),
        agent_learn: Optional[BasePolicy] = None,
        agent_opponent: Optional[BasePolicy] = None,
) -> None:
    env = DummyVectorEnv([lambda: get_env(render_mode=args.render_mode, args=args)])
    # env = get_env(render_mode=args.render_mode, args=args) # Throws error because collector looks for length, could just override though since I'm using my own collector
    policy, optim, agents = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent
    )
    policy.eval()
    policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)

    collector = ManualPolicyCollector(policy, env, exploration_noise=True) # Collector for CPU actions

    pettingzoo_env = env.workers[0].env.env # DummyVectorEnv -> Tianshou PettingZoo Wrapper -> PettingZoo Env
    manual_policy = gobblet_v1.ManualPolicy(pettingzoo_env) # Gobblet keyboard input requires access to raw_env (uses functions from board)

    # Get the first move from the CPU (human goes second))
    if args.player == 1:
        result = collector.collect(n_step=1, render=args.render)

    # Get the first move from the player
    else:
        observation = {"observation": collector.data.obs.obs,
                       "action_mask": collector.data.obs.mask}  # PettingZoo expects a dict with this format
        action = manual_policy(observation, pettingzoo_env.agents[0])

        result = collector.collect_result(action=action.reshape(1), render=args.render)

    while not (collector.data.terminated or collector.data.truncated):
        agent_id = collector.data.obs.agent_id
        # If it is the players turn and there are less than 2 CPU players (at least one human player)
        if agent_id == pettingzoo_env.agents[args.player]:
            # action_mask = collector.data.obs.mask[0]
            # action = np.random.choice(np.arange(len(action_mask)), p=action_mask / np.sum(action_mask))
            observation = {"observation": collector.data.obs.obs,
                            "action_mask": collector.data.obs.mask} # PettingZoo expects a dict with this format
            action = manual_policy(observation, agent_id)

            result = collector.collect_result(action=action.reshape(1), render=args.render)
        else:
            result = collector.collect(n_step=1, render=args.render)

    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, args.agent_id - 1].mean()}, length: {lens.mean()}")

if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = get_args()
    result, agent = train_agent(args)
    if args.cpu_players == 2:

        watch(args, agent)
    else:
        play(args, agent)
