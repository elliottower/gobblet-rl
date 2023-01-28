from gobblet import gobblet_v0
import argparse
import numpy as np


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render_mode", type=str, default="human", help="options: human, human_full, ANSI"
    )
    parser.add_argument(
        "--agent_type", type=str, default="random", help="options: random"
    )
    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = get_args()

    env = gobblet_v0.env(render_mode=args.render_mode, debug=True)
    env.reset()
    turn = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination:
            print(f"Termination ({agent})")
            env.step(None)
        elif truncation:
            print("Truncated")
        else:
            if args.agent_type == "random":
                action = env.action_space(agent).sample()
            if args.agent_type == "random_admissable":
                action_mask = observation.action_mask
                action = np.random.choice(np.arange(len(action_mask)), p=action_mask / np.sum(action_mask))
            env.step(action)

