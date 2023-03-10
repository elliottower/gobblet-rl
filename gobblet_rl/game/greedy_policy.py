from typing import Any, Optional

import numpy as np

from gobblet_rl.game.board import Board


class GreedyGobbletPolicy:
    def __init__(
        self,
        depth: Optional[int] = 2,
        seed: Optional[int] = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.board = None
        self.depth = depth
        self.rng = np.random.default_rng()
        self.prev_actions = {i: [] for i in range(2)}

    def compute_actions_rllib(self, obs_batch):
        observations = obs_batch["observation"]
        observations = observations.reshape(
            observations.shape[0], 3, 3, -1
        )  # Infer observation dimension and batch size
        masks = obs_batch["action_mask"]
        actions = []
        for i in range(len(observations)):
            act = self.compute_action(observations[i], masks[i])
            actions.append(act)
        return actions

    def compute_action_tianshou(self, obs):
        mask = obs.mask
        obs = obs.obs if hasattr(obs, "obs") else obs  #
        return self.compute_action(obs, mask)

    def compute_action(
        self,
        obs,
        mask,
    ) -> np.ndarray:
        obs_player = obs[..., :6]  # Index the last dimension
        board_player = np.array(
            [
                (i + 1) * obs_player[..., i] + (i + 2) * obs_player[..., i + 1]
                for i in range(0, 6, 2)
            ]
        )  # Reshapes [3,3,6] to [3,3,3]
        obs_opponent = obs[..., 6:12]
        board_opponent = np.array(
            [
                (i + 1) * obs_opponent[..., i] + (i + 2) * obs_opponent[..., i + 1]
                for i in range(0, 6, 2)
            ]
        )
        board = np.where(board_player > board_opponent, board_player, -board_opponent)

        agent_index = obs[
            ..., 12
        ].max()  # Thirteenth layer of obs encodes agent_index (all zeros or all ones)
        opponent_index = (
            1 - agent_index
        )  # If agent index is 1, we want 0; if agent index is 0, we want 1

        # If we are playing as the second agent, we need to adjust the representation of the board to reflect that
        if agent_index == 1:
            board = -board

        self.board = Board()
        self.board.squares = board.flatten().copy()

        # Map agent index to winner value that the board will return (agent idx: [0,1], winner_vals: [1,-1])
        winner_values = [1, -1]

        legal_actions = mask.flatten().nonzero()[0]
        actions_depth1 = list(
            legal_actions
        )  # Initialize the same as legal actions, then remove ones that cause a loss
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
                elif (
                    results[action] == winner_values[opponent_index]
                ):  # Loss for our agent
                    if len(actions_depth1) > 1:
                        actions_depth1.remove(action)
                    else:
                        break  # If there is nothing we can do to prevent them from winning, we have to do one of the moves

        if self.depth > 1:
            # Depth 2 (only explore cases where the immediate action doesn't result in a win or loss)
            for action in [key for key, value in results.items() if value == 0]:
                # if self.board.is_legal(agent_index=agent_index, action=action):
                depth1_board = Board()
                depth1_board.squares = self.board.squares.copy()
                depth1_board.play_turn(agent_index=agent_index, action=action)

                # Check what the opponent can do after this potential move
                legal_actions_depth2 = [
                    act
                    for act in range(len(mask.flatten()))
                    if depth1_board.is_legal(agent_index=opponent_index, action=act)
                ]

                results_depth2 = {}
                for action_depth2 in legal_actions_depth2:
                    depth2_board = Board()
                    depth2_board.squares = depth1_board.squares.copy()
                    depth2_board.play_turn(
                        agent_index=opponent_index, action=action_depth2
                    )

                    results_depth2[action_depth2] = depth2_board.check_for_winner()

                    # If the opponent can win in the next move, we don't want to do this action
                    if (
                        results_depth2[action_depth2] == winner_values[opponent_index]
                    ):  # Check if it's possible for the opponent to win next turn after
                        if len(actions_depth1) > 1:
                            if action in actions_depth1:
                                actions_depth1.remove(action)
                        else:
                            break  # If there is nothing we can do to prevent them from winning, we have to pick one

                        # If we can put our piece in the place that the opponent would have gone to win the game, do that
                        # BUT this might miss us from winning the game ourselves in blcoking them, so don't exit the loop
                        # So we only do it if there aren't deterministic wins already chosen (and we don't break, keep looking after)
                        if self.board.is_legal(action_depth2, agent_index=agent_index):
                            if chosen_action is None:
                                chosen_action = action_depth2

                # Depth 2: if this move sets the opponent up with only moves that win the game for us, pick this move
                if all(
                    winner == winner_values[agent_index]
                    for winner in results_depth2.values()
                ):
                    chosen_action = action
                    break
                # Depth 2: if this move blocks the opponent from winning, pick this if we cannot find any guaranteed wins
                if all(
                    winner != winner_values[opponent_index]
                    for winner in results_depth2.values()
                ):
                    chosen_action = action  # BLOCKING ACTION

                    # Depth 3: Given that we block the opponent this way, can we win the next turn?
                    if self.depth == 3:
                        depth1_board = Board()
                        depth1_board.squares = self.board.squares.copy()
                        depth1_board.play_turn(agent_index=agent_index, action=action)

                        # Search over depth 2 actions where the opponent doesn't win
                        for action_depth2 in [
                            key for key, value in results_depth2.items() if value == 0
                        ]:
                            depth2_board = Board()
                            depth2_board.squares = depth1_board.squares.copy()
                            depth2_board.play_turn(
                                agent_index=agent_index, action=action_depth2
                            )

                            legal_actions_depth3 = [
                                act
                                for act in range(len(mask.flatten()))
                                if depth2_board.is_legal(
                                    agent_index=agent_index, action=act
                                )
                            ]
                            actions_depth3 = list(legal_actions_depth3)
                            res_depth3 = {}

                            # Search over all depth 3 actions which we can do
                            for act_depth3 in legal_actions_depth3:
                                depth3_board = Board()
                                depth3_board.squares = depth2_board.squares.copy()
                                depth3_board.play_turn(
                                    agent_index=agent_index, action=action
                                )

                                # If we can win next turn, do it
                                res_depth3[act_depth3] = depth3_board.check_for_winner()
                                if (
                                    res_depth3[act_depth3] == winner_values[agent_index]
                                ):  # Win for our agent
                                    chosen_action = action  # If we can win in depth 3, then we know this blocking action is good
                                    break
                                elif (
                                    res_depth3[act_depth3]
                                    == winner_values[opponent_index]
                                ):  # Loss for our agent
                                    if len(actions_depth3) > 1:
                                        if action in actions_depth3:
                                            actions_depth3.remove(action)
                                    else:
                                        break  # If there is nothing we can do to prevent them from winning in depth3, we have to do one of the moves

        # If we have not selected an action, or have already done it in the last 3 turns, choose randomly
        if (
            chosen_action is None
            or chosen_action in self.prev_actions[agent_index][-3:]
        ):
            # Choose randomly between possible actions:
            # chosen_action = self.rng.choice(actions_depth1)
            chosen_action = np.random.choice(actions_depth1)
            # print(f"Choosing randomly between possible actions: {actions_depth1} --> {chosen_action}")
        self.prev_actions[agent_index].append(chosen_action)
        act = np.array(chosen_action)
        return act
