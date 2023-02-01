import pygame
from gobblet.game.utils import GIFRecorder
import numpy as np
import argparse

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=None, help="Set random seed manually (will only affect CPU agents)"
    )
    parser.add_argument(
        "--cpu-players", type=int, default=1, choices=[0, 1, 2], help="Number of CPU players (options: 0, 1, 2)"
    )
    parser.add_argument(
        "--player", type=int, default=0, choices=[0,1], help="Choose which player to play as: red = 0, yellow = 1"
    )
    parser.add_argument(
        "--screen-width", type=int, default=640, help="Width of pygame screen in pixels"
    )

    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

if __name__ == "__main__":
    from gobblet import gobblet_v1

    args = get_args()

    clock = pygame.time.Clock()

    if args.seed is not None:
        np.random.seed(args.seed)

    env = gobblet_v1.env(render_mode="human", args=args)
    env.reset()

    recorder = GIFRecorder(width=1000, height=1000)

    env.render()  # need to render the environment before pygame can take user input

    # Record the first frame (empty board)
    recorder.capture_frame(env.unwrapped.screen)

    manual_policy = gobblet_v1.ManualPolicy(env, recorder=recorder)

    for agent in env.agent_iter():
        clock.tick(env.metadata["render_fps"])

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            print(f"Agent: ({agent}), Reward: {reward}, info: {info}")

            recorder.end_recording()

            env.step(None)
            continue

        if agent == manual_policy.agent and args.cpu_players < 2:
                action = manual_policy(observation, agent)
        else:
            action_mask = observation['action_mask']
            action = np.random.choice(np.arange(len(action_mask)), p=action_mask / np.sum(action_mask))

        env.step(action)

        env.render()