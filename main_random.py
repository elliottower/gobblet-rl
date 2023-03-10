import asyncio
import sys

sys.path.append("modules")

import time  # noqa: E402

import numpy as np  # noqa: E402 F401

from gobblet_rl import gobblet_v1  # noqa: E402

PLAYER = 0
DEPTH = 2
RENDER_MODE = "human"
RENDER_DELAY = 0.1


async def main() -> None:
    env = gobblet_v1.env(render_mode="human", args=None)
    env.reset()
    env.render()  # need to render the environment before pygame can take user input

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            print(f"Agent: ({agent}), Reward: {reward}, info: {info}")
            break

        action_mask = observation["action_mask"]
        action = np.random.choice(
            np.arange(len(action_mask)), p=action_mask / np.sum(action_mask)
        )

        # Wait .5 seconds between moves so the user can follow the sequence of moves
        time.sleep(0.5)
        env.step(action)

        await asyncio.sleep(0)  # Very important, and keep it 0


if __name__ == "__main__":
    asyncio.run(main())
