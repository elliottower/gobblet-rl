import pettingzoo
import pettingzoo.test
import pytest
import numpy as np
from gobblet import gobblet_v1

# Note: raw_env is required in order to test the board state, as env() only allows observations
@pytest.fixture(scope="function")
def env():
    env = gobblet_v1.raw_env()
    env.reset()
    yield env
    env.close()


def test_reset(env):
    "Verify that reset() executes without error"
    env.reset()


def test_reset_starting(env):
    "Verify that reset() sets the board state to the correct starting position"

    assert (
        (env.board.squares == np.zeros(27)).all()
    )


def test_api(env):
    "Test the env using PettingZoo's API test function"
    pettingzoo.test.api_test(env, num_cycles=10, verbose_progress=False)


def test_parallel_api(env):
    env = gobblet_v1.parallel_env()
    pettingzoo.test.parallel_api_test(env, num_cycles=1000)


def test_seed(env):
    pettingzoo.test.seed_test(gobblet_v1.env)


def test_seed_raw(env):
    pettingzoo.test.seed_test(gobblet_v1.raw_env)


@pytest.mark.skip(
    reason="parallel envs are not currently used, therefore there is no max_cycles argument "
)
def test_max_cycles(env):
    pettingzoo.test.max_cycles_test(env)


# Note: this test sometimes fails due to empty possible actions list, re-run if it fails
def test_performance_benchmark(env):
    "Run PettingZoo performance benchmark on the env"
    env = gobblet_v1.env()
    pettingzoo.test.performance_benchmark(env)


def test_save_obs(env):
    pettingzoo.test.test_save_obs(env)


def test_render(env):
    "Verify that render() executes without error for human-readable output"
    pettingzoo.test.render_test(gobblet_v1.raw_env)


def test_render_human(env):
    "Verify that render() executes without error for human-readable output"
    env.render()
