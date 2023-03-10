from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
from numpy import ndarray
from tianshou.data import Batch
from tianshou.policy import BasePolicy

from gobblet_rl.game.greedy_policy import GreedyGobbletPolicy


class GreedyPolicy(BasePolicy):
    """Greedy Policy.

    Basic greedy policy which checks if a move results in a victory, and if it sets the opponent up to win (or lose) in the next turn.
    The depth argument controls the agent's search depth (default of 2 is a balance between computational efficiency and optimal play)
     * depth = 1: Agent considers moves which it can use to directly win
     * depth = 2: Agent also considers moves it can take to block the opponent from winning next turn
     * depth = 3: Agent also considers moves which set it up to win in two moves: no matter what opponents does in retaliation (unblockable wins)
    """

    def __init__(
        self,
        depth: Optional[int] = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.board = None
        self.depth = depth
        self.policy = GreedyGobbletPolicy()

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
                    batch_single[input].agent_id = (
                        batch_single[input].agent_id[b, ...].reshape(1)
                    )
                act = self.forward_unbatched(batch=batch_single)  # array(1)
                acts.append(act)  # [ array(1), array(2), ... array(10) ]
            acts = np.array(acts)
            return Batch(
                act=acts
            )  # Batch( act: [ array(1), array(2), ... array(10) ] )
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
    ) -> ndarray:
        obs = batch[input]
        act = self.policy.compute_action_tianshou(obs)
        return act

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Since a random agent learns nothing, it returns an empty dict."""
        return {}
