# noqa
"""
# Gobblet

```{figure} classic_tictactoe.gif # TODO: gif of gobblet?
:width: 140px
:name: Gobblet
```

This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.

| Import             | `from gobblet import gobblet_v1`   |
|--------------------|------------------------------------|
| Actions            | Discrete                           |
| Parallel API       | Yes                                |
| Manual Control     | No                                 |
| Agents             | `agents= ['player_1', 'player_2']` |
| Agents             | 2                                  |
| Action Shape       | Discrete(53)                       |
| Action Values      | Discrete(53)                       |
| Observation Shape  | (3,3,15)                           |
| Observation Values | [0,1]                              |


Gobblet is a simple turn based strategy game based on tic-tac-toe, where 2 players, white and black, take turns placing pieces on a 3 x 3 grid.
They can choose from three sizes of pieces: small, medium and large. Each player has two pieces of each size, making six total.
Players can either place a piece in an empty space, "gobble" an opponent's piece with a larger piece, or move an already placed piece either to an empty space or to gobble an opponent's piece.
The first player to place 3 of their pieces in a horizontal, vertical, or diagonal line is the winner.

### Observation Space

The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.

Inspired by AlphaZero's observation space for Chess, the main observation space is a 3x3 image representing the board. It has channels representing:

* Channel 0: Is black or white
* Channels 1 - 12: One channel for each piece type and player combination. For example, channels 1-2 represent white small pieces, 3-4 white medium, and 5-6 white large pieces.
An index of this channel is set to 1 if a white piece is in the corresponding spot on the game board, otherwise it is set to 0.
* Channel 13: A move clock counting up to the 50 move rule. Represented by a single channel where the *n* th element in the flattened channel is set if there has been *n* moves
* Channel 14: All ones to help neural networks find board edges in padded convolutions
* Channel 15: represents whether a position has been seen before (whether a position is a 2-fold repetition)
# TODO: do we need channels 0 and 13-15? Testing without at first

Like AlphaZero, the board is always oriented towards the current agent (the currant agent's king starts on the 1st row). In other words, the two players are looking at mirror images of the board, not the same board.

Unlike AlphaZero, the observation space does not stack the observations previous moves by default. This can be accomplished using the `frame_stacking` argument of our wrapper.

#### Legal Actions Mask

The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.

### Action Space

Actions range from 0 - 53, representing possible positions on the board to place each piece. We count moving already placed pieces the same as placing them regularly.
Actions 0-8 represent positions on the board to place the first small piece.
Actions 9-17 represent positions on the board to place the second small piece.
Actions 18-26 represent positions on the board to place the first medium piece.
Actions 27-35 represent positions on the board to place the second medium piece.
Actions 36-44 represent positions on the board to place the first large piece.
Actions 45-53 represent positions on the board to place the second large piece.

The cells in the board are indexed as follows:

 ```
0 | 3 | 6
_________

1 | 4 | 7
_________

2 | 5 | 8
 ```

To determine the position from an action, we take the number modulo 9, resulting in a number 0-8
To determine the piece from an action i, we use floor division by 9 (i // 9), resulting in a number 0-5, where 1-2 represent small pieces, 3-4 represent medium pieces, and 5-6 represent large pieces.
To determine the size of a given piece p (1-6), we use floor division by 2 (p + 1 // 2) resulting in a number 1-3

### Rewards

### Rewards

| Winner | Loser |
| :----: | :---: |
| +1     | -1    |

If the game ends in a draw, both players will receive a reward of 0.

### Version History

* v3: Fixed bug in arbitrary calls to observe() (1.8.0)
* v2: Legal action mask in observation replaced illegal move list in infos (1.5.0)
* v1: Bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
* v0: Initial versions release (1.0.0)

"""

import gymnasium
import numpy as np
from gymnasium import spaces
import pygame
import os

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from .board import Board
from .utils import get_image, load_chip, load_chip_preview


def env(render_mode=None, args=None):
    env = raw_env(render_mode=render_mode, args=args)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)

class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "text", "text_full"],
        "name": "gobblet_v1",
        "is_parallelizable": True,
        "render_fps": 1,
    }

    def __init__(self, render_mode=None, args=None):
        super().__init__()
        self.board = Board()
        self.board_size = 3 # Will need to make a separate file for 4x4

        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {i: spaces.Discrete(54) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(3, 3, 12), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(54,), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {"legal_moves": list(range(0, 9))} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.render_mode = render_mode
        self.debug = args.debug
        self.screen_width = args.screen_width
        self.screen_height = args.screen_width # Ensure dimensions are the same to prevent scaling issues
        self.screen = None

    # Key
    # ----
    # blank space = 0
    # agent 0 = 1
    # agent 1 = 2
    # An observation is list of lists, where each list represents a row
    #
    # [[0,0,2]
    #  [1,2,1]
    #  [2,1,0]]
    def observe(self, agent): #TODO: test this
        board = self.board.squares.reshape(3, 3, 3)
        if self.agents.index(agent) == 1:
            board = board * -1 # Swap the signs on the board for the two different agents

        # Represent observations in the same way as pettingzoo.chess: specific channel for each color piece (e.g., two for each white small piece)
        layers = []
        for i in range(1, 7):
            layers.append(board[(i - 1) // 2] == i) # 3x3 array with an entry of 1 for squares with each white piece (1, ..., 6)

        for i in range(1, 7):
            layers.append(board[(i - 1) // 2] == -i)  # 3x3 array with an entry of 1 for squares with each black piece (-1, ..., -6)
        observation = np.stack(layers, axis=2).astype(np.int8)
        legal_moves = self._legal_moves() if agent == self.agent_selection else []

        action_mask = np.zeros(54, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _legal_moves(self):
        legal_moves = []
        for action in range(54):
            if self.board.is_legal(action):
                legal_moves.append(action)
        return legal_moves

    # is a value from 0 to 8 indicating position to move on 3x3 board
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        # check if input action is a valid move (0 == empty spot)
        # assert self.board.is_legal(action), "played illegal move"
        if not self.board.is_legal(action):
            print("piece: ", self.board.get_piece_from_action(action))
            print("piece_size: ", self.board.get_piece_size_from_action(action))
            print("pos: ", self.board.get_pos_from_action(action))
            print("--ERROR-- ILLEGAL MOVE")
        # play turn
        self.board.play_turn(self.agents.index(self.agent_selection), action)

        next_agent = self._agent_selector.next()

        if self.board.check_game_over():
            winner = self.board.check_for_winner()
            if winner == 0: # NOTE: don't think ties are possible in gobblet
                # tie
                pass
            elif winner == 1:
                # agent 0 won
                self.rewards[self.agents[0]] += 1
                self.rewards[self.agents[1]] -= 1
            else:
                # agent 1 won
                self.rewards[self.agents[1]] += 1
                self.rewards[self.agents[0]] -= 1

            # once either play wins or there is a draw, game over, both players are done
            self.terminations = {i: True for i in self.agents}

        # Switch selection to next agents
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_agent

        self._accumulate_rewards()
        self.turn += 1
        self.action = action
        if self.render_mode == "human" or "text" or "human_full":
            self.render()


    def reset(self, seed=None, return_info=False, options=None):
        # reset environment
        self.board = Board()

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()
        self.turn = 0
        self.action = -1

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        def getSymbolFull(input):
            if input == 0:
                return "- "
            if input > 0:
                return "+{}".format(int(input))
            else:
                return "{}".format(int(input))

        def getSymbol(input):
            if input == 0:
                return "- "
            if input > 0:
                return "+{}".format(int((input + 1) // 2))
            else:
                return "{}".format(int((input) // 2))

        if self.debug:
            self.board.print_pieces()
        if self.render_mode == "text" or self.debug:
            pos = self.action % 9
            piece = (self.action // 9) + 1
            piece = (piece + 1) // 2
            print(f"TURN: {self.turn}, AGENT: {self.agent_selection}, ACTION: {self.action}, POSITION: {pos}, PIECE: {piece}")
            board = list(map(getSymbol, self.board.get_flatboard()))
            print(" " * 7 + "|" + " " * 7 + "|" + " " * 7)
            print(f"  {board[0]}   " + "|" + f"   {board[3]}  " + "|" + f"   {board[6]}  ")
            print("_" * 7 + "|" + "_" * 7 + "|" + "_" * 7)

            print(" " * 7 + "|" + " " * 7 + "|" + " " * 7)
            print(f"  {board[1]}   " + "|" + f"   {board[4]}  " + "|" + f"   {board[7]}  ")
            print("_" * 7 + "|" + "_" * 7 + "|" + "_" * 7)

            print(" " * 7 + "|" + " " * 7 + "|" + " " * 7)
            print(f"  {board[2]}   " + "|" + f"   {board[5]}  " + "|" + f"   {board[8]}  ")
            print(" " * 7 + "|" + " " * 7 + "|" + " " * 7)
            print()

        elif self.render_mode == "text_full":
            pos = self.action % 9
            piece = (self.action // 9) + 1
            print(f"TURN: {self.turn}, AGENT: {self.agent_selection}, ACTION: {self.action}, POSITION: {pos}, PIECE: {piece}")
            board = list(map(getSymbolFull, self.board.squares))
            print(" " * 9 + "SMALL" + " " * 9 + "  " +
                  " " * 10 + "MED" + " " * 10 + "  " +
                  " " * 9 + "LARGE" + " " * 9 + "  ")
            top= " " * 7 + "|" + " " * 7 + "|" + " " * 7
            bottom = "_" * 7 + "|" + "_" * 7 + "|" + "_" * 7
            top1= f"  {board[0]}   " + "|" + f"   {board[3]}  " + "|" + f"   {board[6]}  "
            top2 = f"  {board[9]}   " + "|" + f"   {board[12]}  " + "|" + f"   {board[15]}  "
            top3 = f"  {board[18]}   " + "|" + f"   {board[21]}  " + "|" + f"   {board[24]}  "
            print(top + "  " + top + "  " + top)
            print(top1 + "  " + top2 + "  " + top3)
            print(bottom + "  " + bottom + "  " + bottom)

            mid1 = f"  {board[1]}   " + "|" + f"   {board[4]}  " + "|" + f"   {board[7]}  "
            mid2 = f"  {board[10]}   " + "|" + f"   {board[13]}  " + "|" + f"   {board[16]}  "
            mid3 = f"  {board[19]}   " + "|" + f"   {board[22]}  " + "|" + f"   {board[25]}  "
            print(top + "  " + top + "  " + top)
            print(mid1 + "  " + mid2 + "  " + mid3)
            print(bottom + "  " + bottom + "  " + bottom)

            bot1 = f"  {board[2]}   " + "|" + f"   {board[5]}  " + "|" + f"   {board[8]}  "
            bot2 = f"  {board[9+2]}   " + "|" + f"   {board[9+5]}  " + "|" + f"   {board[9+8]}  "
            bot3 = f"  {board[18+2]}   " + "|" + f"   {board[18+5]}  " + "|" + f"   {board[18+8]}  "
            print(top + "  " + top + "  " + top)
            print(bot1 + "  " + bot2 + "  " + bot3)
            print(top + "  " + top + "  " + top)
            print()

        elif self.render_mode == "human":
            # Adapted from PettingZoo connect_four.py (keep width equal to height or things will not scale properly)
            if self.render_mode == "human":
                if self.screen is None:
                    pygame.init()
                    self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.event.get()
            elif self.screen is None:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

            # Load and scale all the necessary images
            # Settled on these numbers by trial & error (happens to make things fit nicely)
            tile_size = (self.screen_width * ((47 - 7) / 47)) / 3
            scale_large = 9 / 13
            scale_med = 6 / 13
            scale_small = 4 / 13
            red = {}
            red[3] = load_chip(tile_size, "GobbletLargeRed.png", scale_large)
            red[2] = load_chip(tile_size, "GobbletMedRed.png", scale_med)
            red[1] = load_chip(tile_size, "GobbletSmallRed.png", scale_small)

            yellow = {}
            yellow[3] = load_chip(tile_size, "GobbletLargeYellow.png", scale_large)
            yellow[2] = load_chip(tile_size, "GobbletMedYellow.png", scale_med)
            yellow[1] = load_chip(tile_size, "GobbletSmallYellow.png", scale_small)

            self.preview = {}
            self.preview["player_1"] = {}
            self.preview["player_1"][3] = load_chip_preview(tile_size, "GobbletLargeRedPreview.png", scale_large)
            self.preview["player_1"][2] = load_chip_preview(tile_size, "GobbletMedRedPreview.png", scale_med)
            self.preview["player_1"][1] = load_chip_preview(tile_size, "GobbletSmallRedPreview.png", scale_small)

            self.preview["player_2"] = {}
            self.preview["player_2"][3] = load_chip_preview(tile_size, "GobbletLargeYellowPreview.png", scale_large)
            self.preview["player_2"][2] = load_chip_preview(tile_size, "GobbletMedYellowPreview.png", scale_med)
            self.preview["player_2"][1] = load_chip_preview(tile_size, "GobbletSmallYellowPreview.png", scale_small)

            preview_chips = {self.agents[0]: self.preview["player_1"], self.agents[1]: self.preview["player_1"]}

            board_img = get_image(os.path.join("img", "GobbletBoard3x3.png"))
            board_img = pygame.transform.scale(
                board_img, ((int(self.screen_width)), int(self.screen_height))
            )

            self.screen.blit(board_img, (0, 0))

            offset = (self.screen_width * ((9+4) / 47)) # Piece is 9px wide, gap between pieces 4px, total width is 47px
            offset_side = (self.screen_width * (6 / 47)) - 1 # Distance from the side of the board to the first piece is 6px
            offset_centering = offset * 1/3 + \
                               (5 * self.screen_width/1000 if self.screen_width > 500 else 8 * self.screen_width/1000)
            # Extra 5px fixed alignment issues at 1000x1000, but ratio needs to be higher for lower res (trial & error)

            # Blit the chips and their positions
            for i in range(9):
                for j in range(1, 4):
                    if self.board.squares[i + 9 * (j - 1)] == 2 * j - 1 or \
                        self.board.squares[i + 9 * (j - 1)] == 2 * j: # small pieces (1,2), medium pieces (3,4), large pieces (5,6)
                        self.screen.blit(
                            red[j],
                            red[j].get_rect(center =
                                                (int(i / 3) * (offset) + offset_side + offset_centering,
                                                     (i % 3) * (offset) + offset_side + offset_centering
                                                 )
                                            )
                        )
                    if self.board.squares[i + 9 * (j - 1)] == -1 * (2  * j - 1) or\
                        self.board.squares[i + 9 * (j - 1)] == -1 * (2 * j):
                        self.screen.blit(
                            yellow[j],
                            yellow[j].get_rect(center =
                                                (int(i / 3) * (offset) + offset_side + offset_centering,
                                                     (i % 3) * (offset) + offset_side + offset_centering
                                                 )
                                            )
                        )

            # Blit the preview chips and their positions
            for i in range(9):
                for j in range(1, 4):
                    if self.board.squares_preview[i + 9 * (j - 1)] == 1:
                        self.screen.blit(
                            self.preview["player_1"][j],
                            self.preview["player_1"][j].get_rect(center =
                                                (int(i / 3) * (offset) + offset_side + offset_centering,
                                                     (i % 3) * (offset) + offset_side + offset_centering
                                                 )
                                            )
                        )
                    if self.board.squares_preview[i + 9 * (j - 1)] == -1:
                        self.screen.blit(
                            self.preview["player_2"][j],
                            self.preview["player_2"][j].get_rect(center =
                                                (int(i / 3) * (offset) + offset_side + offset_centering,
                                                     (i % 3) * (offset) + offset_side + offset_centering
                                                 )
                                            )
                        )

            pygame.display.update()
        elif self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.quit()
            self.screen = None
