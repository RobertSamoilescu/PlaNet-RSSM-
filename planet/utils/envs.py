import gym
from typing import Any, Dict, Optional
from planet.utils.wrappers import RepeatActionWrapper, GymPixelWrapper, ImagePreprocessorWrapper



class BaseEnv:
    def __init__(self, config: Dict[str, Any]) -> None:
        pass
    
    def reset(self) -> Any:
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
        self.env = gym.make(config["id"])
        self.env.reset(seed=config.get("seed"))
        self.env = RepeatActionWrapper(self.env, skip=config.get("skip", 4))
        
    def reset(self) -> Any:
        return self.env.reset()
    
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
                "render_kwargs", 
                {'width': 64, 'height': 64}
            ),
        )
        self.env = RepeatActionWrapper(self.env, skip=config.get("skip", 4))
        self.env = ImagePreprocessorWrapper(self.env)
                
                
    def reset(self) -> Any:
        return self.env.reset()
    
    def step(self, action: Any) -> Any:
        return self.env.step(action)
    
    

def make_env(config: Dict[str, Any]) -> Any:
    if config["env_type"] == "gym":
        return GymEnv(config)
    elif config["env_type"] == "dm_control":
        return DMControlEnv(config)
    else:
        raise ValueError("Invalid environment type")