from gobblet import gobblet_v1
import argparse
import numpy as np
import time
import pygame
import sys
from utils import GIFRecorder

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-mode", type=str, default="human", help="Select rendering mode (options: human, text, text_full)"
    )
    parser.add_argument(
        "--agent-type", type=str, default="random_admissible", help="Select agent type for CPU agents (options: random, random_admissible)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Set random seed manually (will only affect CPU agents)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Flag to display extra debugging information (text only)"
    )
    parser.add_argument(
        "--no-cpu", action="store_true", help="Flag to disable CPU players and play as both teams"
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Flag to enable CPU only games (no human input)"
    )
    parser.add_argument(
        "--player", type=int, default=0, help="Choose which player to play as: red = 0, yellow = 1"
    )
    parser.add_argument(
        "--save-gif", action="store_true", help="Save screen recording of gameplay to gif (requires imagemick: https://imagemagick.org)"
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

    env = gobblet_v1.env(render_mode=args.render_mode, args=args)
    if args.seed is not None:
        env.reset(seed=args.seed)
        np.random.seed(args.seed)
    else:
        env.reset()
    turn = 0
    if args.save_gif:
        recorder = GIFRecorder(width=1000, height=1000)
    env.render()  # need to render the environment before pygame can take user input
    if args.save_gif:
        recorder.capture_frame(env.unwrapped.screen)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination:
            print(f"Termination ({agent}), Reward: {reward}, info: {info}")
            if args.save_gif:
                recorder.end_recording()
            env.step(None)
        elif truncation:
            print("Truncated")
        else:
            if args.agent_type == "random":
                time.sleep(.1)
                action = env.action_space(agent).sample()
            if args.agent_type == "random_admissible":
                time.sleep(.1)
                action_mask = observation['action_mask']
                action = np.random.choice(np.arange(len(action_mask)), p=action_mask / np.sum(action_mask))
            if (agent == env.agents[args.player] or args.no_cpu) and not args.cpu_only:
                picked_up = False
                picked_up_pos = -1
                piece_cycle = 0
                piece_size_selected = 0
                while True:
                    event = pygame.event.wait()
                    if event.type == pygame.QUIT:
                        if args.save_gif:
                            recorder.end_recording()
                        pygame.quit()
                        pygame.display.quit()
                        sys.exit()
                    mousex, mousey = pygame.mouse.get_pos()
                    width, height = pygame.display.get_surface().get_size()
                    pos_x = 0
                    if 0 * width/1000 <= mousex < 360 * width/1000:
                        pos_x = 0
                    elif 360 * width/1000 <= mousex < 640 * width/1000:
                        pos_x = 1
                    elif 640 * width/1000 <= mousex < 1000 * width/1000:
                        pos_x = 2
                    pos_y = 0
                    if 0 * height/1000 <= mousey < 360 * height/1000:
                        pos_y = 0
                    elif 360 * height/1000 <= mousey < 640 * height/1000:
                        pos_y= 1
                    elif 640 * height/1000 <= mousey < 1000 * height/1000:
                        pos_y = 2
                    pos = pos_y + 3 * (pos_x)

                    agent_multiplier = 1 if agent == env.agents[0] else -1

                    movable_pieces = [ i // 9 for i in observation["action_mask"].nonzero() ]
                    placed_pieces = env.unwrapped.board.squares[env.unwrapped.board.squares.nonzero()]
                    placed_pieces_agent = [a for a in placed_pieces if np.sign(a) == agent_multiplier]
                    placed_pieces_agent_abs = [abs(p) for p in placed_pieces_agent]
                    pieces = np.arange(1, 7)
                    unplaced = [p for p in pieces if p not in placed_pieces_agent_abs]
                    flatboard = env.unwrapped.board.get_flatboard()

                    # Selected piece size persists as we loop through events, allowing the user to cycle through sizes
                    if piece_size_selected > 0:
                        piece = piece
                    else:
                        piece = unplaced[-1]
                        piece_size_selected = (piece + 1) // 2

                    # Read keyboard input
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            flag = True
                            piece_cycle += 1
                        else:
                            if event.key == pygame.K_1 and not picked_up: # User inputs for pieces of size 1 (indices 1 and 2)
                                piece_cycle = 0
                                if 1 in unplaced:
                                    piece = 1
                                    piece_size_selected = 1
                                elif 2 in unplaced:
                                    piece = 2
                                    piece_size_selected = 1
                                else: piece = -1
                            elif event.key == pygame.K_2 and not picked_up:
                                piece_cycle = 0
                                if 3 in unplaced:
                                    piece = 3
                                    piece_size_selected = 2
                                elif 4 in unplaced:
                                    piece = 4
                                    piece_size_selected = 2
                                else: piece = -1
                            elif event.key == pygame.K_3 and not picked_up:
                                piece_cycle = 0
                                if 5 in unplaced:
                                    piece = 5
                                    piece_size_selected = 3
                                elif 6 in unplaced:
                                    piece = 6
                                    piece_size_selected = 3
                                else: piece = -1
                    # Don't render a preview if both pieces of a given size have been placed
                    if piece != -1:
                        piece_size = ( piece + 1 ) // 2
                        # Don't render a preview if both pieces of a given size have been placed
                        if piece_cycle:
                            if piece_cycle and not picked_up:
                                cycle_choices = np.unique(
                                    [(p + 1) // 2 for p in unplaced])  # Transform [1,2,3,4,5,6] to [1,2,3)
                                piece_size = cycle_choices[
                                    (np.amax(cycle_choices) - (piece_cycle + 1)) % len(cycle_choices)]  # Cycle from largest to smallest

                        # If the hovered over position means placing a picked up piece in the same spot, mark it as illegal
                        if pos == picked_up_pos:
                            action_prev = -1

                        # Get the action from the position the mouse cursor is currently hovering over
                        action_prev = env.unwrapped.board.get_action(pos, piece_size, env.agents.index(agent))

                    # If the hovered over position means placing a picked up piece in the same spot, mark it as illegal
                    if pos == picked_up_pos:
                        action_prev = -1

                    # Clear previously previewed moves
                    env.unwrapped.board.squares_preview[:] = 0
                    if action_prev != -1:
                        if not env.unwrapped.board.is_legal(action_prev): # If this action is illegal
                            action_prev = -1
                        else:
                            env.unwrapped.board.squares_preview[pos + 9 * (piece_size-1)] = agent_multiplier # Preview this position

                    # Update the display with the previewed move
                    env.render()
                    pygame.display.update()
                    if args.save_gif:
                        recorder.capture_frame(env.unwrapped.screen)

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        # Pick up a piece (only able to if it has already been placed, and is not currently picked up)
                        if flatboard[pos] in placed_pieces_agent and not picked_up :
                            piece_size_on_board = (abs(flatboard[pos]) + 1) // 2
                            piece_to_pick_up = int(flatboard[pos])
                            piece = abs(piece_to_pick_up)

                            # If the piece size selected is larger, clicking here should self-gobble the smaller piece
                            if piece_size_on_board >= piece_size_selected:
                                # Can only pick up a piece if there is a legal move to place it, other than where it was before
                                if not all(observation["action_mask"][9 * (piece - 1): 9 * piece] == 0):
                                    picked_up = True
                                    picked_up_pos = pos
                                    # Remove a placed piece
                                    piece_size_selected = (piece + 1) // 2

                                    index = np.where(env.unwrapped.board.squares == piece_to_pick_up)[0][0]
                                    env.unwrapped.board.squares[index] = 0

                                    # Set the only possible actions to be moving this piece to a new square
                                    observation["action_mask"][pos + 9 * (piece-1)] = 0 # TODO: check if this is already zero
                                    observation["action_mask"][:9 * (piece - 1)] = 0 # Zero out all the possible actions
                                    observation["action_mask"][9 * (piece):] = 0

                        # If we are not picking a piece up, then try to place a piece, if it is legal to do so
                        else:
                            if action_prev != -1:
                                env.unwrapped.board.squares_preview[pos + 9 * (piece_size-1)] = 0
                                action = pos + 9 * (piece - 1)
                                break
            time.sleep(.1)
            env.render()
            if args.save_gif:
                recorder.capture_frame(env.unwrapped.screen)
            env.step(action)