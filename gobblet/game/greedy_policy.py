from typing import Any, Dict, Optional, Union
from copy import deepcopy
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
        depth: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.board = None
        self.depth = depth

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
        full_batch = deepcopy(batch)
        if len(batch) > 1:
            acts = []
            for b in range(len(batch)):
                batch_single = deepcopy(full_batch)
                if len(batch_single[input].agent_id) > 1:
                    batch_single[input].obs = batch_single[input].obs[b, ...]
                    batch_single[input].mask = batch_single[input].mask[b, ...]
                    batch_single[input].agent_id = batch_single[input].agent_id[b, ...].reshape(1)
                act = self.forward_unbatched(batch=batch_single)  # array(1)
                acts.append(act) # [ array(1), array(2), ... array(10) ]
            acts = np.array(acts)
            return Batch(act=acts)  # Batch( act: [ array(1), array(2), ... array(10) ] )
        else:
            batch_single = deepcopy(full_batch)
            act = self.forward_unbatched(batch=batch_single)
            acts = np.array(act).reshape(1)
            return Batch(act=acts)

    def forward_unbatched(
        self,
        batch: Batch,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        obs = batch[input]
        obs_depth2 = obs.obs if hasattr(obs, "obs") else obs  #

        obs_player = obs_depth2[..., :6]  # Index the last dimension
        board_player = np.array([(i + 1) * obs_player[..., i] + (i + 2) * obs_player[..., i + 1] for i in
                                 range(0, 6, 2)])  # Reshapes [3,3,6] to [3,3,3]
        obs_opponent = obs_depth2[..., 6:12]
        board_opponent = np.array(
            [(i + 1) * obs_opponent[..., i] + (i + 2) * obs_opponent[..., i + 1] for i in range(0, 6, 2)])
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

        legal_actions = obs.mask.flatten().nonzero()[0]
        actions_depth1 = list(legal_actions) # Initialize the same as legal actions, then remove ones that cause a loss
        chosen_action = None

        results = {}
        # Depth 1
        for action in legal_actions:
            if self.board.is_legal(agent_index=agent_index, action=action):
                depth1_board = Board()
                depth1_board.squares = self.board.squares.copy()
                depth1_board.play_turn(agent_index=agent_index, action=action)

                # If we can win next turn, do it
                results[action] = depth1_board.check_for_winner()
                if results[action] == winner_values[agent_index]:  # Win for our agent
                    chosen_action = action
                    break
                elif results[action] == winner_values[opponent_index]:  # Loss for our agent
                    if len(actions_depth1) > 1:
                        actions_depth1.remove(action)
                    else:
                        break  # If there is nothing we can do to prevent them from winning, we have to do one of the moves

        if self.depth > 1:
            # Depth 2 (only explore cases where the immediate action doesn't result in a win or loss)
            for action in [ key for key, value in results.items() if value == 0 ]:
                # if self.board.is_legal(agent_index=agent_index, action=action):
                depth1_board = Board()
                depth1_board.squares = self.board.squares.copy()
                depth1_board.play_turn(agent_index=agent_index, action=action)

                # Check what the opponent can do after this potential move
                legal_actions_depth2 = [act for act in range(len(obs.mask.flatten())) if
                                      depth1_board.is_legal(agent_index=opponent_index, action=act)]

                results_depth2 = {}
                for action_depth2 in legal_actions_depth2:
                    depth2_board = Board()
                    depth2_board.squares = depth1_board.squares.copy()
                    depth2_board.play_turn(agent_index=opponent_index, action=action_depth2)

                    results_depth2[action_depth2] = depth2_board.check_for_winner()

                    # If the opponent can win in the next move, we don't want to do this action
                    if results_depth2[action_depth2] == winner_values[
                        opponent_index]:  # Check if it's possible for the opponent to win next turn after
                        if len(actions_depth1) > 1:
                            if action in actions_depth1:
                                actions_depth1.remove(action)
                        else:
                            break  # If there is nothing we can do to prevent them from winning, we have to pick one

                        # If we can put our piece in the place that the opponent would have gone to win the game, do that
                        # BUT this might miss us from winning the game ourselves in blcoking them, so don't exit the loop
                        # So we only do it if there aren't deterministic wins already chosen (and we don't break, keep looking after)
                        if self.board.is_legal(action_depth2, agent_index=agent_index):
                            if chosen_action == None:
                                chosen_action = action_depth2

                # Depth 2: if this move sets the opponent up with only moves that win the game for us, pick this move
                if all(winner == winner_values[agent_index] for winner in results_depth2.values()):
                    chosen_action = action
                    break
                # Depth 2: if this move blocks the opponent from winning, pick this if we cannot find any guaranteed wins
                if all(winner != winner_values[opponent_index] for winner in results_depth2.values()):
                    chosen_action = action # BLOCKING ACTION

                    # Depth 3: Given that we block the opponent this way, can we win the next turn?
                    if self.depth == 3:
                        depth1_board = Board()
                        depth1_board.squares = self.board.squares.copy()
                        depth1_board.play_turn(agent_index=agent_index, action=action)

                        # Search over depth 2 actions where the opponent doesn't win
                        for action_depth2 in [ key for key, value in results_depth2.items() if value == 0 ]:
                            depth2_board = Board()
                            depth2_board.squares = depth1_board.squares.copy()
                            depth2_board.play_turn(agent_index=agent_index, action=action_depth2)

                            legal_actions_depth3 = [act for act in range(len(obs.mask.flatten())) if
                                                depth2_board.is_legal(agent_index=agent_index, action=act)]
                            actions_depth3 = list(legal_actions_depth3)
                            results_depth3 = {}

                            # Search over all depth 3 actions which we can do
                            for action_depth3 in legal_actions_depth3:
                                depth3_board = Board()
                                depth3_board.squares = depth2_board.squares.copy()
                                depth3_board.play_turn(agent_index=agent_index, action=action)

                                # If we can win next turn, do it
                                results_depth3[action_depth3] = depth3_board.check_for_winner()
                                if results_depth3[action_depth3] == winner_values[agent_index]:  # Win for our agent
                                    chosen_action = action # If we can win in depth 3, then we know this blocking action is good
                                    break
                                elif results_depth3[action_depth3] == winner_values[opponent_index]:  # Loss for our agent
                                    if len(actions_depth3) > 1:
                                        actions_depth3.remove(action)
                                    else:
                                        break  # If there is nothing we can do to prevent them from winning in depth3, we have to do one of the moves

        if chosen_action is None:
            # Choose randomly between possible actions:
            chosen_action = np.random.choice(actions_depth1)
            # print(f"Choosing randomly between possible actions: {actions_depth1} --> {chosen_action}")
        act = np.array(chosen_action)
        return act

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Since a random agent learns nothing, it returns an empty dict."""
        return {}