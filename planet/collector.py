import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Dict, Optional, List, Tuple

from planet.dataset.env_objects import EnvStep
from planet.models.utils import set_models_eval
from planet.planning.planner import latent_planning


@torch.no_grad()
def collect_episode(
    env: Any,
    models: Dict[str, nn.Module],
    rnn_hidden_size: int,
    action_size: int,
    max_episode_length: int,
    H: int = 15,
    I: int = 10,
    J: int = 1_000,
    K: int = 100,
    action_min: float = -1.0,
    action_max: float = 1.0,
    action_noise: Optional[float] = None,
    device: str = "cuda",
) -> Tuple[List[EnvStep], float]:
    """Collect an episode using the world model.

    :param env: The environment to collect the episode
    :param model: Dictionary of models
    :param rnn_hidden_size: The size of the rnn hidden state
    :param H: The number of latent samples
    :param I: The number of action sequences
    :param J: The number of action samples
    :param K: The number of planning steps
    :param action_min: The minimum action value
    :param action_max: The maximum action value
    :param action_noise: The action noise
    :param device: The device to run the model
    :return: A tuple of list of environment steps and the total reward
    """
    set_models_eval(models)

    # reset environment
    sequence, episode_reward = [], 0
    obs_np, _ = env.reset()

    # initialize hidden state
    hidden_state = torch.zeros(1, rnn_hidden_size).to(device)

    for _ in tqdm(range(max_episode_length), desc="Collecting"):
        # compute posterior distribution based on the current observation
        obs = torch.from_numpy(obs_np).float().to(device)
        posterior_dist = models["enc_model"](
            hidden_state=hidden_state,
            observation=obs.unsqueeze(0),
        )

        # compute action using latent planning
        action = latent_planning(
            H=H,
            I=I,
            J=J,
            K=K,
            hidden_state=hidden_state,
            current_state_belief=posterior_dist,
            deterministic_state_model=models["det_state_model"],
            stochastic_state_model=models["stoch_state_model"],
            reward_model=models["reward_obs_model"],
            action_size=action_size,
            action_min=action_min,
            action_max=action_max,
        )

        # add exploration noise
        if action_noise is not None:
            action += torch.randn_like(action) * action_noise
            action = torch.clamp(action, action_min, action_max)

        # take action in the environment
        action_np = action.cpu().numpy()
        next_obs_np, reward, terminated, truncated, _ = env.step(action_np)

        # update episode reward and
        # add step to the sequence
        episode_reward += reward
        done = 1 if terminated or truncated else 0
        sequence.append(
            EnvStep(
                observation=torch.from_numpy(obs_np),
                action=torch.from_numpy(action_np),
                reward=reward,
                done=done,
            )
        )

        if done == 1:
            break

        # update observation
        obs_np = next_obs_np

        # update hidden state
        hidden_state = models["det_state_model"](
            hidden_state=hidden_state,
            state=posterior_dist.sample(),
            action=action.unsqueeze(0),
        )

    return sequence, episode_reward
