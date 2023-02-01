import glob
import os
from typing import Tuple

import ray.tune
from ray import init
from ray.rllib.agents import ppo
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from gobblet import gobblet_v1
from gobblet.models.action_mask_model import TorchActionMaskModel
from gobblet.game.utils import get_project_root

torch, nn = try_import_torch()


def prepare_train() -> Tuple[ppo.PPOTrainer, PettingZooEnv]:
    env_name = "pettingzoo_gobblet"

    # get the Pettingzoo env
    def env_creator():
        env = env = gobblet_v1.env(render_mode=None, debug=False)
        return env

    register_env(env_name, lambda config: PettingZooEnv(env_creator()))
    ModelCatalog.register_custom_model("pa_model2", TorchActionMaskModel)
    # wrap the pettingzoo env in MultiAgent RLLib
    env = PettingZooEnv(env_creator())
    agents = ["player_1", "player_2"]
    custom_config = {
        "env": env_name,
        "model": {
            "custom_model": "pa_model2",
        },
        "framework": "torch",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(torch.cuda.device_count()),
        "num_workers": 1, #os.cpu_count() - 1,
        "multiagent": {
            "policies": {
                name: (None, env.observation_space, env.action_space, {})
                for name in agents
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
    }

    # get trainer

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(custom_config)

    trainer = ppo.PPOTrainer(config=ppo_config)

    return trainer, env, ppo_config


def train(trainer, max_steps=2e6):
    # run manual training loop and print results after each iteration
    iters = 0
    while True:
        iters += 1
        result = trainer.train()
        print(pretty_print(result))
        # stop training if the target train steps or reward are reached
        if result["timesteps_total"] >= max_steps:
            print(
                "training done, because max_steps"
                f"{max_steps} {result['timesteps_total']} reached"
            )
            break
    # manual test loop
    return trainer


def train_ray(ppo_config, timesteps_total: int = 10):
    analysis = ray.tune.run(
        ppo.PPOTrainer,
        config=ppo_config,
        local_dir=os.path.join(get_project_root(), "models"),
        stop={"timesteps_total": timesteps_total},
        checkpoint_at_end=True,
    )
    return analysis


def load_ray(path, ppo_config):
    """
    Load a trained RLlib agent from the specified path.
    Call this before testing a trained agent.
    :param path:
        Path pointing to the agent's saved checkpoint (only used for RLlib agents)
    :param ppo_config:
        dict config
    """
    trainer = ppo.PPOTrainer(config=ppo_config)
    trainer.restore(path)
    return trainer


def sample_trainer(trainer, env):
    print("Finished training. Running manual test/inference loop.")
    obs = env.reset()
    done = {"__all__": False}
    # run one iteration until done

    for i in range(10000):
        if done["__all__"]:
            print("game done")
            break
        # get agent from current observation
        agent = list(obs.keys())[0]

        # format observation dict
        print(obs)
        obs = obs[agent]
        env.render()

        # get deterministic action
        # trainer.compute_single_action(obs, policy_id=agent)
        policy = trainer.get_policy(policy_id=agent)
        action_exploration_policy, _, action_info = policy.compute_single_action(obs)
        logits = action_info["action_dist_inputs"]
        action = logits.argmax()
        env.render()
        print("agent ", agent, " action ", action)
        obs, reward, done, _ = env.step({agent: action})
        # observations contain original observations and the action mask
        # print(f"Obs: {obs}, Action: {action}, done: {done}")

    env.render()
    print(env.env.rewards)


def tune_training_loop(timesteps_total=10000):
    """train trainer and sample"""
    trainer, env, ppo_config = prepare_train()

    # train trainer
    analysis = train_ray(ppo_config, timesteps_total=timesteps_total)
    # reload the checkpoint
    last_chpt_path = analysis._checkpoints[-1]["local_dir"]
    checkpoint_file = glob.glob(
        os.path.join(last_chpt_path, "**", "checkpoint-*"), recursive=True
    )[0]
    trainer_trained = load_ray(checkpoint_file, ppo_config)

    # sample trainer
    sample_trainer(trainer_trained, env)


def manual_training_loop(timesteps_total=10000):
    """train trainer and sample"""

    trainer, env, ppo_config = prepare_train()
    trainer_trained = train(trainer, max_steps=timesteps_total)

    sample_trainer(trainer_trained, env)


if __name__ == "__main__":
    init(local_mode=True)
    tune_training_loop()