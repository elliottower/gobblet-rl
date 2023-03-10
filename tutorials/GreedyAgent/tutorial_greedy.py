import time

import numpy as np

from gobblet_rl import gobblet_v1  # noqa: E402

PLAYER = 0
DEPTH = 2
RENDER_MODE = "human"

if __name__ == "__main__":
    env = gobblet_v1.env(render_mode="human", args=None)

    greedy_policy = gobblet_v1.GreedyGobbletPolicy(depth=DEPTH)

    # Render 3 games between greedy agents
    for _ in range(3):
        env.reset()
        env.render()  # need to render the environment before pygame can take user input

        iter = 0

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                env.render()
                time.sleep(1)
                print(f"Agent: ({agent}), Reward: {reward}, info: {info}")
                break

            if iter < 2:
                # Randomize the first action for variety (games can be repeated otherwise)
                action_mask = observation["action_mask"]
                action = np.random.choice(
                    np.arange(len(action_mask)), p=action_mask / np.sum(action_mask)
                )
                # Wait 1 second between moves so the user can follow the sequence of moves
                time.sleep(1)

            else:
                action = greedy_policy.compute_action(
                    observation["observation"], observation["action_mask"]
                )
                # Wait 1 second between moves so the user can follow the sequence of moves
                time.sleep(1)

            env.step(action)

            iter += 1
