import gym
import numpy as np
from typing import Any, Dict
from planet.utils.wrappers import (
    RepeatActionWrapper,
    GymPixelWrapper,
    ImagePreprocessorWrapper,
)


class BaseEnv:
    def __init__(self, config: Dict[str, Any]) -> None:
        pass

    def reset(self) -> Any:
        raise NotImplementedError

    def render(self) -> Any:
        raise NotImplementedError

    def step(self, action: Any) -> Any:
        raise NotImplementedError

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


class GymEnv(BaseEnv):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.env = gym.make(config["id"], **config.get("kwargs", {}))
        self.env.reset(seed=config.get("seed"))
        self.env = RepeatActionWrapper(self.env, skip=config.get("skip", 4))

    def reset(self) -> Any:
        return self.env.reset()

    def render(self) -> Any:
        return self.env.render()

    def step(self, action: Any) -> Any:
        return self.env.step(action)


class DMControlEnv(BaseEnv):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.env = GymPixelWrapper(
            domain_name=config["domain_name"],
            task_name=config["task_name"],
            task_kwargs=config.get("task_kwargs", {}),
            render_kwargs=config.get(
                "render_kwargs", {"width": 64, "height": 64, "camera_id": 0}
            ),
        )
        self.env = RepeatActionWrapper(self.env, skip=config.get("skip", 4))  # type: ignore[arg-type, assignment]  # noqa: E501
        self.env = ImagePreprocessorWrapper(self.env)  # type: ignore[arg-type, assignment]  # noqa: E501
        self.last_obs = None

    def reset(self) -> Any:
        self.last_obs, info = self.env.reset()
        return self.last_obs, info

    def render(self) -> Any:
        return ((self.last_obs.transpose(1, 2, 0) + 0.5) * 255).astype(
            np.uint8
        )

    def step(self, action: Any) -> Any:
        self.last_obs, reward, terminate, truncated, info = self.env.step(
            action
        )
        return self.last_obs, reward, terminate, truncated, info


def make_env(config: Dict[str, Any]) -> Any:
    if config["env_type"] == "gym":
        return GymEnv(config)
    elif config["env_type"] == "dm_control":
        return DMControlEnv(config)
    else:
        raise ValueError("Invalid environment type")
