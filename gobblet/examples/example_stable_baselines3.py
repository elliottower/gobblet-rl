from stable_baselines3 import PPO

from stable_baselines3.common.env_checker import check_env
from pettingzoo.test import parallel_api_test
import supersuit as ss

from gobblet import gobblet_v1

env = gobblet_v1.parallel_env(render_mode=None, args=None)

env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')

experiment_name = "test"
rollout_fragment_length = 50
seed = 0

model = PPO("MlpPolicy", env, tensorboard_log=f"/tmp/{experiment_name}", verbose=3, gamma=0.95,
    n_steps=rollout_fragment_length, ent_coef=0.01,
    learning_rate=5e-5, vf_coef=1, max_grad_norm=0.9, gae_lambda=1.0, n_epochs=30, clip_range=0.3,
    batch_size=150, seed=seed)
# train_timesteps = 100 * 1000
train_timesteps = 1000
model.learn(total_timesteps=train_timesteps)
model.save(f"policy_gobblet_{train_timesteps}")