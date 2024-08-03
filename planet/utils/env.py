import gym
import numpy as np


def n_step_interaction(
    env: gym.Env,
    n: int,
    action: np.ndarray,
):
    """
    Perform n-step interaction with the environment.

    :param env: The environment.
    :param n: The number of steps to interact with the environment.
    :param action: The action to take in the environment.
    """
    reward_sum = 0

    for _ in range(n):
        _obs, _reward, terminated, truncated, info = env.step(action)
        reward_sum += _reward

        if terminated or truncated:
            break

    return _obs, reward_sum, terminated, truncated, info
