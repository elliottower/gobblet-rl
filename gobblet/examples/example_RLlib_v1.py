import gymnasium
import gymnasium.core
import numpy as np
from pettingzoo.utils import wrappers
from ray.rllib.agents import ppo
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from gobblet import gobblet_v1

# # Note: gymnasium.core.Wrapper will cause the observation to be a string 'player_1', rather than the dict
# class ReshapedEnv(gym.core.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#         # Number of actions for each agent (for redefining action space method below)
#         self.n = {}
#
#         # Flatten observation (RLlib will change model type from fully connected to CNN for obs space with 3+ dims)
#         for p in env.possible_agents:
#             old = self.observation_space(p)['observation']
#             new_shape = (old.shape[0] * old.shape[1], old.shape[2])
#             self.observation_space(p)['observation'] = gymnasium.spaces.Box(low=old.low.reshape(new_shape), high=old.high.reshape(new_shape), shape=new_shape, dtype=old.dtype)
#
#             # Number of actions for this agent
#             self.n[p] = env.action_space(env.possible_agents[0]).n
#
#     # Convert to gym.spaces.Discrete (RLlib model catalog expects action space to be gym.spaces.Discrete. rather than gymnasium)
#     def action_space(self, agent):
#         return gym.spaces.Discrete(self.n[agent])
#
#     # Override `observation` to custom process the original observation
#     # coming from the env.
#     def observation(self, observation):
#         # If the observation consists of a dict with keys ['player_1', 'player_2'] (expected for parallel env)
#         if isinstance(observation, dict):
#             if self.env.agents ==  list(observation.keys()):
#                 for key in observation.keys():
#                     observation[key]["observation"] = observation[key]["observation"].reshape(9, 12)
#             elif "observation" in observation.keys():
#                 observation["observation"] = observation["observation"].reshape(9, 12)
#         else:
#             observation = observation.reshape(9, 12)
#         return observation


class FlattenedEnv(wrappers.base.BaseWrapper):
    def __init__(self, env):
        # # Number of actions for each agent (for redefining action space method below)
        # self.n = {}
        super().__init__(env)

        # Flatten observation (RLlib will change model type from fully connected to CNN for obs space with 3+ dims)
        for p in env.possible_agents:
            old = self.observation_space(p)["observation"]
            new_shape = np.prod(old.shape).reshape(1)
            self.observation_space(p)["observation"] = gymnasium.spaces.Box(
                low=old.low.reshape(new_shape),
                high=old.high.reshape(new_shape),
                shape=new_shape,
                dtype=old.dtype,
            )

    def observe(self, agent):
        observation = self.env.observe(agent)
        print("observation: ", observation)

        new_shape = self.observation_space(self.env.possible_agents[0])[
            "observation"
        ].shape
        # print(new_shape)
        # If the observation consists of a dict with keys ['player_1', 'player_2'] (expected for parallel env)

        if isinstance(observation, dict):
            if self.env.agents == list(observation.keys()):
                for key in observation.keys():
                    observation[key]["observation"] = observation[key][
                        "observation"
                    ].reshape(new_shape)
            elif "observation" in observation.keys():
                observation["observation"] = observation["observation"].reshape(
                    new_shape
                )
        else:
            observation = observation.reshape(new_shape)
        print("flattened observation: ", observation)
        return observation

    def __str__(self):
        return str(self.env)


def env_creator(args):
    env = gobblet_v1.parallel_env()
    env = FlattenedEnv(env)
    # env = ReshapedEnv(env)
    # env = ss.reshape_v0(env, (9, 12))
    return env


if __name__ == "__main__":
    env_name = "gobblet_pettingzoo"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    test_env = ParallelPettingZooEnv(env_creator({}))

    # # Convert obs space and act space to gym (what RLlib expects)
    # obs_space = test_env.observation_space
    # obs_space = obs_space["observation"]
    # obs_space = gym.spaces.Box(obs_space.low, obs_space.high, obs_space.shape, obs_space.dtype)
    # act_space = test_env.action_space
    # act_space = gym.spaces.Discrete(act_space.n)

    agents = ["player_1", "player_2"]
    custom_config = {
        "env": env_name,
        "framework": "torch",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # "num_gpus": int(torch.cuda.device_count()),
        "num_gpus": 0,
        "num_workers": 1,  # os.cpu_count() - 1,
        "multiagent": {
            "policies": {
                name: (
                    PolicySpec(
                        policy_class=None,  # infer automatically from Algorithm
                        observation_space=test_env.observation_space,  # infer automatically from env
                        action_space=test_env.action_space,  # infer automatically from env
                        config={
                            "gamma": 0.85
                        },  # use main config plus <- this override here
                    )
                )
                for name in agents
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
    }

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(custom_config)

    trainer = ppo.PPOTrainer(config=ppo_config)

    result = trainer.train()

    #  # Below code might define policy wrong, was copied from somewhere else
    #
    # def gen_policy(i):
    #     config = {
    #         "gamma": 0.99,
    #     }
    #     return (None, test_env.observation_space, test_env.action_space, config)
    #
    # policies = {"policy_0": gen_policy(0)}
    #
    # policy_ids = list(policies.keys())
    #
    # tune.run(
    #     "PPO",
    #     name="PPO",
    #     stop={"timesteps_total": 5000000},
    #     checkpoint_freq=10,
    #     local_dir="~/ray_results/"+env_name,
    #     config={
    #         # Environment specific
    #         "env": env_name,
    #         "no_done_at_end": True,
    #         "num_gpus": 0,
    #         "num_workers": 2,
    #         "num_envs_per_worker": 1,
    #         "compress_observations": False,
    #         "batch_mode": 'truncate_episodes',
    #         "clip_rewards": False,
    #         "vf_clip_param": 500.0,
    #         "entropy_coeff": 0.01,
    #         # effective batch_size: train_batch_size * num_agents_in_each_environment [5, 10]
    #         # see https://github.com/ray-project/ray/issues/4628
    #         "train_batch_size": 1000,  # 5000
    #         "rollout_fragment_length": 50,  # 100
    #         "sgd_minibatch_size": 100,  # 500
    #         "vf_share_layers": False
    #         },
    # )
