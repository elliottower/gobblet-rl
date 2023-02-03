from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import BasePolicy

from gobblet.game.board import Board

class GreedyPolicy(BasePolicy):
    """
    Basic greedy policy which checks if a move results in a victory, and if it sets the opponent up to win (or lose) in the next turn
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.board = None

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :return: A :class:`~tianshou.data.Batch` with "act" key, containing
            the greedy action.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs

        obs_player = obs_next[..., :6]  # Index the last dimension
        board_player = np.array([ (i+1) * obs_player[..., i] + (i+2) * obs_player[..., i + 1] for i in range(0, 6, 2)]) # Reshapes [3,3,6] to [3,3,3]
        obs_opponent = obs_next[..., 6:12]
        board_opponent = np.array([ (i+1) * obs_opponent[..., i] + (i+2) * obs_opponent[..., i + 1] for i in range(0, 6, 2)])
        board = np.where(board_player > board_opponent, board_player, -board_opponent)

        agents = ["player_1", "player_2"]
        agent_index = agents.index(obs.agent_id[0])
        opponent_index = 1 - agent_index  # If agent index is 1, we want 0; if agent index is 0, we want 1

        # If we are playing as the second agent, we need to adjust the representation of the board to reflect that
        # TODO: it would be cleaner to add a thirteenth channel that explicitly encodes the player whose turn it is
        # avoids issues of the wrong agent index (easier debugging) and enforces turn order in a stricter way
        if agent_index == 1:
            board = -board

        self.board = Board()
        self.board.squares = board.flatten().copy()

        # Map agent index to winner value that the board will return (agent idx: [0,1], winner_vals: [1,-1])
        winner_values = [1, -1]

        legal_actions = obs.mask.nonzero()[1]
        next_actions = list(legal_actions) # Initialize the same as legal actions, then remove ones that cause a loss
        chosen_action = None


        for action in legal_actions:
            if self.board.is_legal(agent_index=agent_index, action=action):
                next_board = Board()
                next_board.squares = self.board.squares.copy()
                next_board.play_turn(agent_index=agent_index, action=action)

                # If we can win next turn, do it
                winner = next_board.check_for_winner()
                if winner == winner_values[agent_index]: # Win for our agent
                    chosen_action = action
                    break
                elif winner == winner_values[opponent_index]: # Loss for our agent
                    if len(next_actions) > 1:
                        next_actions.remove(action)
                    else:
                        break  # If there is nothing we can do to prevent them from winning, we have to do one of the moves
                    print()
                else:
                    # Otherwise, check what the opponent can do after this potential move
                    legal_actions_next = [next_act for next_act in range(obs.mask.shape[-1]) if next_board.is_legal(agent_index=opponent_index, action=next_act)]
                    # print("legal actions next", legal_actions_next)

                    winner_next = {}
                    for action_next in legal_actions_next:
                        next_next_board = Board()
                        next_next_board.squares = next_board.squares.copy()
                        next_next_board.play_turn(agent_index=opponent_index, action=action_next)

                        winner_next[action_next] = next_next_board.check_for_winner()

                        # If the opponent can win in the next move, we don't want to do this action
                        if winner_next[action_next] == winner_values[opponent_index]: # Check if it's possible for the opponent to win next turn after
                            if len(next_actions) > 1:
                                if action in next_actions:
                                    next_actions.remove(action)
                            else:
                                break # If there is nothing we can do to prevent them from winning, we have to pick one

                            # # If we can put our piece in the place that the opponent would have gone to win the game, do that
                            # # BUT this might miss us from winning the game ourselves in blcoking them, so don't exit the loop
                            if self.board.is_legal(action_next, agent_index=agent_index):
                                chosen_action = action_next
                            # break

                    # Pick the move if it prevents the opponent from winning no matter what he does after our move
                    if all(winner != winner_values[opponent_index] for winner in winner_next.values()):
                        chosen_action = action
                        break

        if chosen_action is None:
            chosen_action = np.random.choice(next_actions)
            print(f"Choosing randomly between possible actions: {next_actions} --> {chosen_action}")
        act = np.array(chosen_action).reshape(1)
        return Batch(act=act)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Since a random agent learns nothing, it returns an empty dict."""
        return {}