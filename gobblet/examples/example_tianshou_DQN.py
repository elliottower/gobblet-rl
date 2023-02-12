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
import gymnasium
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
from gobblet.game.utils import GIFRecorder
from gobblet.game.greedy_policy import GreedyPolicy
import time


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4) # TODO: Changing this to 1e-5 for some reason makes it pause after 3 or 4 epochs
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
    parser.add_argument("--debug", action="store_true", help="Flag to enable to print extra debugging info")
    parser.add_argument("--self_play", action="store_true", help="Flag to enable training via self-play (as opposed to fixed opponent)")
    parser.add_argument("--self_play_generations", type=int, default=5, help="Number of generations of self-play agents to train (each generation can have multiple epochs of training)")
    parser.add_argument("--self_play_greedy", action="store_true", help="Flag to have self-play train against a greedy agent for the first generation")
    parser.add_argument("--cpu-players", type=int, default=2, choices=[1, 2], help="Number of CPU players (options: 1, 2)")
    parser.add_argument("--player", type=int, default=0, choices=[0,1], help="Choose which player to play as: red = 0, yellow = 1")
    parser.add_argument("--record", action="store_true", help="Flag to save a recording of the game (game.gif)")
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
        if isinstance(env.observation_space, gym.spaces.Dict) or isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = (
        observation_space.shape or observation_space.n
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
            # Create a new network with the same shape
            net_opponent = Net(
                args.state_shape,
                args.action_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)
            agent_opponent = DQNPolicy(
                net_opponent,
                optim,
                args.gamma,
                args.n_step,
                target_update_freq=args.target_update_freq,
            )
        elif args.opponent_path:
            agent_opponent = deepcopy(agent_learn)
            agent_opponent.load_state_dict(torch.load(args.opponent_path))
        else:
            # agent_opponent = RandomPolicy()
            agent_opponent = GreedyPolicy() # Greedy policy is a difficult opponent, should yeild much better results than random

    if args.agent_id == 1:
        agents = [agent_learn, agent_opponent]
    else:
        agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def get_env(render_mode=None, args=None):
    return PettingZooEnv(gobblet_v1.env(render_mode=render_mode, args=args))


def train_selfplay(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_fixed: Optional[BasePolicy] = None,
):
    # always train first agent, start from random policy

    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== environment space =========
    # model
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict) or isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = (
            observation_space.shape or observation_space.n
    )
    args.action_shape = env.action_space.shape or env.action_space.n

    # ======== model setup =========
    # Note: custom models can be specified using the agent_fixed and agent_learn arguments
    if agent_learn is None:
        net = Net(args.state_shape, args.action_shape,
                  hidden_sizes=args.hidden_sizes, device=args.device
                  ).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        agent_learn = DQNPolicy(
            net, optim, args.gamma, args.n_step,
            target_update_freq=args.target_update_freq)

    if agent_fixed is None:
        net_fixed = Net(args.state_shape, args.action_shape,
                        hidden_sizes=args.hidden_sizes, device=args.device
                        ).to(args.device)
        optim_fixed = torch.optim.SGD(net_fixed.parameters(), lr=0)
        agent_fixed = DQNPolicy(
            net_fixed, optim_fixed, args.gamma, args.n_step,
            target_update_freq=args.target_update_freq)

    # Load initial opponent from file
    # path = os.path.join(args.logdir, 'gobblet', 'dqn', 'policy.pth')
    # agent_fixed.load_state_dict(torch.load(path))

    # ======== agent setup =========
    agents = [agent_learn, agent_fixed]
    policy = MultiAgentPolicyManager(agents, env)
    agents_list = list(policy.policies.keys())


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
    log_path = os.path.join(args.logdir, "gobblet", "dqn-selfplay")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, "gobblet", "dqn-selfplay", "policy.pth"
            )
        torch.save(
            policy.policies[agents_list[args.agent_id - 1]].state_dict(), model_save_path
        )

    def stop_fn(mean_rewards):
        return mean_rewards >= args.win_rate

    def train_fn(epoch, env_step):
        policy.policies[agents_list[0]].set_eps(args.eps_train)

        if hasattr(policy.policies[agents_list[1]], "set_eps"):
            policy.policies[agents_list[1]].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        policy.policies[agents_list[0]].set_eps(args.eps_test)

        if hasattr(policy.policies[agents_list[1]], "set_eps"):
            policy.policies[agents_list[1]].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, 0]


    # Self-play loop
    for i in range(args.self_play_generations):
        result = offpolicy_trainer(
            policy, train_collector, test_collector, args.epoch,
            args.step_per_epoch, args.step_per_collect, args.test_num,
            args.batch_size, train_fn=train_fn, test_fn=test_fn,
            stop_fn=stop_fn, save_best_fn=save_best_fn, update_per_step=args.update_per_step,
            logger=logger, test_in_train=False, reward_metric=reward_metric)

        print(f"Launching game between learned agent ({type(policy.policies[agents_list[0]]).__name__}) and fixed agent ({type(policy.policies[agents_list[1]]).__name__}):")
        # Render a single game between the learned policy and fixed policy from last generation
        watch_selfplay(args, policy.policies[agents_list[0]])

        # Set fixed opponent policy as to the current trained policy (updates every epoch)
        policy.policies[agents_list[1]] = deepcopy(policy.policies[agents_list[0]])
        print('--- SELF-PLAY GENERATION: {} ---'.format(i + 1))

        print(f"Launching game between learned agent ({type(policy.policies[agents_list[0]]).__name__}) and itself ({type(policy.policies[agents_list[1]]).__name__}):")
        # Render a single game between the learned policy and itself
        watch_selfplay(args, policy.policies[agents_list[0]])

    model_save_path = os.path.join(args.logdir, 'gobblet', 'dqn-selfplay', 'policy.pth')
    torch.save(policy.policies[agents_list[0]].state_dict(), model_save_path)

    return result, policy.policies[agents_list[0]]


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
    agents_list = list(policy.policies.keys())

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
            policy.policies[agents_list[args.agent_id - 1]].state_dict(), model_save_path
        )

    def stop_fn(mean_rewards):
        return mean_rewards >= args.win_rate

    def train_fn(epoch, env_step):
        policy.policies[agents_list[args.agent_id - 1]].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        policy.policies[agents_list[args.agent_id - 1]].set_eps(args.eps_test)

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
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    return result, policy.policies[agents_list[args.agent_id - 1]]


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
    agents_list = list(policy.policies.keys())

    policy.eval()
    policy.policies[agents_list[args.agent_id - 1]].set_eps(args.eps_test)

    collector = Collector(policy, env, exploration_noise=True)
    pettingzoo_env = env.workers[0].env.env  # DummyVectorEnv -> Tianshou PettingZoo Wrapper -> PettingZoo Env
    if args.record:
        recorder = GIFRecorder()
        recorder.capture_frame(pettingzoo_env.unwrapped.screen)
    else:
        recorder = None

    while pettingzoo_env.agents:
        result = collector.collect(n_step=1, render=args.render)
        if recorder is not None:
            recorder.capture_frame(pettingzoo_env.unwrapped.screen)

        time.sleep(0.25)

        if collector.data.terminated or collector.data.truncated:
            rews, lens = result["rews"], result["lens"]
            print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()} [{type(policy.policies[agents_list[0]]).__name__}]")
            print(f"Final reward: {rews[:, 1].mean()}, length: {lens.mean()} [{type(policy.policies[agents_list[1]]).__name__}]")
            if recorder is not None:
                recorder.end_recording(pettingzoo_env.unwrapped.screen)
                recorder = None

def watch_selfplay(
        args: argparse.Namespace = get_args(),
        agent: Optional[BasePolicy] = None,
) -> None:
    # env = get_env(render_mode=args.render_mode, args=args)
    env = DummyVectorEnv([lambda: get_env(render_mode=args.render_mode, args=args)])
    policy = MultiAgentPolicyManager(policies=[agent, deepcopy(agent)], env=get_env(render_mode=args.render_mode, args=args)) # fixed here
    policy.eval()
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=True)
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
    agents_list = list(policy.policies.keys())
    policy.eval()
    policy.policies[agents_list[args.agent_id - 1]].set_eps(args.eps_test)

    # Experimental: let the CPU agent to continue training (TODO: check if this actually changes things meaningfully)
    # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)

    collector = ManualPolicyCollector(policy, env, exploration_noise=True) # Collector for CPU actions

    pettingzoo_env = env.workers[0].env.env # DummyVectorEnv -> Tianshou PettingZoo Wrapper -> PettingZoo Env

    if args.record:
        recorder = GIFRecorder()
    else:
        recorder = None
    manual_policy = gobblet_v1.ManualPolicy(env=pettingzoo_env, agent_id=args.player, recorder=recorder) # Gobblet keyboard input requires access to raw_env (uses functions from board)
    while pettingzoo_env.agents:
        agent_id = collector.data.obs.agent_id
        # If it is the players turn and there are less than 2 CPU players (at least one human player)
        if agent_id == pettingzoo_env.agents[args.player]:
            observation = {"observation": collector.data.obs.obs.flatten(),
                            "action_mask": collector.data.obs.mask.flatten()} # PettingZoo expects a dict with this format
            action = manual_policy(observation, agent_id)

            result = collector.collect_result(action=action.reshape(1), render=args.render)
        else:
            result = collector.collect(n_step=1, render=args.render)

            if recorder is not None:
                recorder.capture_frame(pettingzoo_env.unwrapped.screen)

        if collector.data.terminated or collector.data.truncated:
            rews, lens = result["rews"], result["lens"]
            print(f"Final reward: {rews[:, args.player].mean()}, length: {lens.mean()} [Human]")
            print(f"Final reward: {rews[:, 1-args.player].mean()}, length: {lens.mean()} [{type(policy.policies[agents[1-args.player]]).__name__}]")
            if recorder is not None:
                recorder.end_recording(pettingzoo_env.unwrapped.screen)
                recorder = None

if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = get_args()
    if args.player == 1:
        args.agent_id = 1 # Ensures trained agent is in the  correct spot

    if args.self_play:
        print("Training agent...")

        # Hard code the first fixed agent to be the greedy policy (after one generation it will be switched to a copy of the learned agent)
        agent_fixed = GreedyPolicy() if args.self_play_greedy else None
        result, agent = train_selfplay(args=args, agent_fixed=agent_fixed) # Hard code the first fixed agent to be the greedy policy

        print("Starting game...")
        watch_selfplay(args, agent)
    else:
        print("Training agent...")
        result, agent = train_agent(args)

        print("Starting game...")
        if args.cpu_players == 2:
            watch(args, agent)
        else:
            play(args, agent)