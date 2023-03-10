import asyncio
import sys
import time

sys.path.append("modules")

import numpy as np  # noqa: E402 F401

from gobblet_rl import gobblet_v1  # noqa: E402

PLAYER = 0
DEPTH = 2
RENDER_MODE = "human"


async def main() -> None:
    env = gobblet_v1.env(render_mode="human", args=None)

    greedy_policy = gobblet_v1.GreedyGobbletPolicy(depth=DEPTH)

    # Continue to display games between greedy agents
    while True:
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

            await asyncio.sleep(0)  # Very important, and keep it 0
            iter += 1


if __name__ == "__main__":
    asyncio.run(main())
