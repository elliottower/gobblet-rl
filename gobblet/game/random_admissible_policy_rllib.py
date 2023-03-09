from typing import List, Optional, Union

import numpy as np
import tree  # pip install dm_tree
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorStructType, TensorType


class RandomAdmissiblePolicy(RandomPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override(RandomPolicy)
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        **kwargs,
    ):
        if "action_mask" in obs_batch.keys():
            action_masks = obs_batch["action_mask"]
            actions = [
                np.random.choice(
                    np.arange(len(action_mask)), p=action_mask / np.sum(action_mask)
                )
                for action_mask in action_masks
            ]
        else:
            obs_batch_size = len(tree.flatten(obs_batch)[0])
            actions = [
                self.action_space_for_sampling.sample() for _ in range(obs_batch_size)
            ]
        return (
            actions,
            [],
            {},
        )
