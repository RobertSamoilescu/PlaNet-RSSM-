import os

os.environ["MUJOCO_GL"] = "egl"

import gym
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import pixels
from planet.utils.image import preprocess_obs


class RepeatActionWrapper(gym.Wrapper):
    """Action repeat wrapper to act same action repeatedly"""

    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0.0

        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class GymPixelWrapper:
    """Wrapper for gym environment"""

    def __init__(self, domain_name: str, task_name: str, task_kwargs: dict, render_kwargs: dict):
        self._env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs)
        self._env = pixels.Wrapper(self._env, render_kwargs=render_kwargs)

    @property
    def observation_space(self):
        obs_spec = self._env.observation_spec()
        return gym.spaces.Box(0, 255, obs_spec["pixels"].shape, dtype=np.uint8)

    @property
    def action_space(self):
        action_spec = self._env.action_spec()
        return gym.spaces.Box(
            action_spec.minimum, action_spec.maximum, dtype=np.float32
        )

    def reset(self):
        time_step = self._env.reset()
        return time_step.observation["pixels"], None

    def step(self, action: np.ndarray):
        time_step = self._env.step(action)
        observation = time_step.observation["pixels"]
        reward = time_step.reward
        terminated = truncated = time_step.last()
        info = {"discount": time_step.discount}
        return observation, reward, terminated, truncated, info


class ImagePreprocessorWrapper(gym.Wrapper):
    """Preprocess the observation image"""

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        shape = env.observation_space.shape
        shape = (shape[-1], *shape[1:])
        self.observation_space = gym.spaces.Box(
            low=-0.5, high=0.5, shape=shape, dtype=np.float32
        )

    def reset(self):
        obs, info = self.env.reset()
        obs = preprocess_obs(obs)
        return obs.transpose(2, 0, 1), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = preprocess_obs(obs).transpose(2, 0, 1)
        return obs, reward, terminated, truncated, info
