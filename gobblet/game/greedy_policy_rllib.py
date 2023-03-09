from typing import (
    List,
    Optional,
    Union,
)

from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorStructType, TensorType
from ray.rllib.examples.policy.random_policy import RandomPolicy

from gobblet.game.greedy_policy import GreedyGobbletPolicy
import numpy as np


class GreedyPolicy(RandomPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = GreedyGobbletPolicy(seed=np.random.randint(1000), depth=1)

    @override(RandomPolicy)
    def compute_actions(
            self,
            obs_batch: Union[List[TensorStructType], TensorStructType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
            prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
            **kwargs
    ):
        actions = self.policy.compute_actions_rllib(obs_batch)
        return (
            actions,
            [],
            {},
        )
