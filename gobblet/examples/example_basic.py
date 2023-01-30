from gobblet import gobblet_v0
import argparse
import numpy as np
import time
import pygame
import sys

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-mode", type=str, default="human", help="options: human, console, console_full"
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
    parser.add_argument(
        "--no-cpu", action="store_true", help="disable CPU players and play as both teams"
    )
    parser.add_argument(
        "--player", type=int, default=0, help="which player to play as"
    )

    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = get_args()

    env = gobblet_v0.env(render_mode=args.render_mode, debug=args.debug)
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
            if agent == env.agents[args.player] or args.no_cpu:
                while True:
                    event = pygame.event.wait()
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        pygame.display.quit()
                        sys.exit()
                    mousex, mousey = pygame.mouse.get_pos()
                    if  50 <= mousex < 220:
                        action = 0
                    elif 220 <= mousex < 390:
                        action = 1
                    elif 390 <= mousex < 560:
                        action = 2
                    elif 560 <= mousex < 730:
                        action = 3
                    elif 730 <= mousex < 900:
                        action = 4
                    elif 900 <= mousex < 1070:
                        action = 5
                    elif 1070 <= mousex < 1240:
                        action = 8
                    piece_size = 1 # hard code to get previews of large pieces
                    env.unwrapped.board.squares_preview[:] = 0
                    env.unwrapped.board.squares_preview[action * piece_size] = 1
                    env.render()
                    pygame.display.update()
                    print(env.unwrapped.board.squares_preview)
                    print(f"pos: {mousex}, {mousey}")
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        env.unwrapped.board.squares_preview[action * piece_size] = 0
                        break


            time.sleep(.1)
            env.step(action)

