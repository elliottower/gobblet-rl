# Adapted from https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/knights_archers_zombies/manual_policy.py
import pygame
from .utils import GIFRecorder
import sys
import numpy as np


class ManualPolicy:
    def __init__(self, env, agent_id: int = 0, recorder: GIFRecorder = None):

        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id]
        self.recorder = recorder

        env.render()  # need to render the environment before pygame can take user input

    def __call__(self, observation, agent):
        recorder = self.recorder
        env = self.env

        picked_up = False
        picked_up_pos = -1
        piece_cycle = 0
        piece_size_selected = 0

        while True:
            event = pygame.event.wait()

            if event.type == pygame.QUIT:
                if recorder is not None:
                    recorder.end_recording(env.unwrapped.screen)
                pygame.quit()
                pygame.display.quit()
                sys.exit()

            ''' GET MOUSE INPUT'''
            mousex, mousey = pygame.mouse.get_pos()
            width, height = pygame.display.get_surface().get_size()
            pos_x = 0
            if 0 * width / 1000 <= mousex < 360 * width / 1000:
                pos_x = 0
            elif 360 * width / 1000 <= mousex < 640 * width / 1000:
                pos_x = 1
            elif 640 * width / 1000 <= mousex < 1000 * width / 1000:
                pos_x = 2
            pos_y = 0
            if 0 * height / 1000 <= mousey < 360 * height / 1000:
                pos_y = 0
            elif 360 * height / 1000 <= mousey < 640 * height / 1000:
                pos_y = 1
            elif 640 * height / 1000 <= mousey < 1000 * height / 1000:
                pos_y = 2
            pos = pos_y + 3 * (pos_x)

            agent_multiplier = 1 if agent == env.agents[0] else -1

            ''' FIND PLACED PIECES '''
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
                if len(unplaced) > 0:
                    piece = unplaced[-1]
                    piece_size_selected = (piece + 1) // 2
                else:
                    piece = -1

            ''' READ KEYBOARD INPUT'''
            if event.type == pygame.KEYDOWN:
                if not picked_up:
                    if event.key == pygame.K_SPACE:   # Cycle through pieces (from largest to smallest)
                        piece_cycle += 1

                        cycle_choices = np.unique(
                            [(p + 1) // 2 for p in unplaced])  # Transform [1,2,3,4,5,6] to [1,2,3)

                        if len(cycle_choices) > 0:
                            piece_size = cycle_choices[(np.amax(cycle_choices) - (piece_cycle + 1)) % len(cycle_choices)]
                        # else:
                        #   piece_size = -1
                        piece_size_selected = piece_size

                        if (piece_size * 2) - 1 in unplaced:  # Check if the first of this piece size is available
                            piece = piece_size * 2 - 1
                        else:
                            piece = piece_size * 2            # Otherwise choose the second of this piece size
                    else:
                        if event.key == pygame.K_1:    # Select piece size 1
                            piece_size_selected = 1
                            if 1 in unplaced:
                                piece = 1
                                piece_cycle = 2
                            elif 2 in unplaced:
                                piece = 2
                                piece_cycle = 2
                            else:
                                piece = -1
                        elif event.key == pygame.K_2:   # Select piece size 2
                            piece_size_selected = 2
                            if 3 in unplaced:
                                piece = 3
                                piece_cycle = 1
                            elif 4 in unplaced:
                                piece = 4
                                piece_cycle = 1
                            else:
                                piece = -1
                        elif event.key == pygame.K_3:   # Select piece size 3
                            piece_size_selected = 3
                            if 5 in unplaced:
                                piece = 5
                                piece_cycle = 0
                            elif 6 in unplaced:
                                piece = 6
                                piece_cycle = 0
                            else:
                                piece = -1

            # Don't render a preview if both pieces of a given size have been placed
            if piece != -1:
                piece_size = (piece + 1) // 2
                # Don't render a preview if both pieces of a given size have been placed

                ''' GET PREVIEW ACTION '''
                # Get the action from the preview (in position the mouse cursor is currently hovering over)
                action_prev = env.unwrapped.board.get_action(pos, piece_size, env.agents.index(agent))

            ''' CLEAR ACTION PREVIEW FOR ILLEGAL MOVES'''
            # If the hovered over position means placing a picked up piece in the same spot, mark it as illegal
            if pos == picked_up_pos or piece == -1:
                action_prev = -1

            ''' CLEAR PREVIOUSLY PREVIEWED MOVES '''
            env.unwrapped.board.squares_preview[:] = 0
            if action_prev != -1:
                if not env.unwrapped.board.is_legal(action_prev, env.agents.index(agent)):  # If this action is illegal
                    action_prev = -1
                else:
                    env.unwrapped.board.squares_preview[
                        pos + 9 * (piece_size - 1)] = agent_multiplier  # Preview this position

            ''' UPDATE DISPLAY with previewed move'''
            env.render()
            pygame.display.update()
            if recorder is not None:
                recorder.capture_frame(env.unwrapped.screen)

            ''' PICK UP / PLACE A PIECE '''
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Pick up a piece (only able to if it has already been placed, and is not currently picked up)
                if flatboard[pos] in placed_pieces_agent and not picked_up:
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
                            observation["action_mask"][pos + 9 * (piece - 1)] = 0  # TODO: check if this is already zero
                            observation["action_mask"][:9 * (piece - 1)] = 0  # Zero out all the possible actions
                            observation["action_mask"][9 * (piece):] = 0

                # Place a piece (if it is legal to do so)
                else:
                    if action_prev != -1:
                        env.unwrapped.board.squares_preview[pos + 9 * (piece_size - 1)] = 0
                        action = pos + 9 * (piece - 1)
                        break
        return np.int32(action)

    @property
    def available_agents(self):
        return self.env.agent_name_mapping