from gobblet import gobblet_v1
import argparse
import numpy as np
import time


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-mode", type=str, default="human", help="options: human, text, text_full"
    )
    parser.add_argument(
        "--agent-type", type=str, default="random_admissible", help="options: random, random_admissible"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for board and policy"
    )
    parser.add_argument(
        "--debug", action="store_true", help="display extra debugging information"
    )

    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = get_args()

    env = gobblet_v1.env(render_mode=args.render_mode, debug=args.debug)
    if args.seed is not None:
        env.reset(seed=args.seed)
        np.random.seed(args.seed)
    else:
        env.reset()
    turn = 0
    env.render()  # need to render the environment before pygame can take user input
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination:
            print(f"Termination ({agent}), Reward: {reward}, info: {info}")
            env.step(None)
        elif truncation:
            print("Truncated")
        else:
            if args.agent_type == "random":
                action = env.action_space(agent).sample()
            if args.agent_type == "random_admissible":
                action_mask = observation['action_mask']
                action = np.random.choice(np.arange(len(action_mask)), p=action_mask / np.sum(action_mask))
            time.sleep(.1)
            env.step(action)