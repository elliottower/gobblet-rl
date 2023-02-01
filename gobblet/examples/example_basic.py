from gobblet import gobblet_v1
import argparse
import numpy as np
import time


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render_mode", type=str, default="human", choices=["human", "rgb_array", "text", "text_full"],
                        help="Choose the rendering mode for the game."
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for board and policy"
    )
    parser.add_argument(
        "--debug", action="store_true", help="display extra debugging information"
    )
    parser.add_argument(
        "--screen-width", type=int, default=640, help="Width of pygame screen in pixels"
    )

    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = get_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    env = gobblet_v1.env(render_mode=args.render_mode, args=args)

    env.reset()

    env.render()  # need to render the environment before pygame can take user input

    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            print(f"Agent: ({agent}), Reward: {reward}, info: {info}")
            env.step(None)

        else:
            action_mask = observation['action_mask']
            action = np.random.choice(np.arange(len(action_mask)), p=action_mask / np.sum(action_mask))

            if args.render_mode == "human":
                time.sleep(5) # Wait .5 seconds between moves so the user can follow the sequence of moves

            env.step(action)