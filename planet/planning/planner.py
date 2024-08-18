import torch
import torch.nn as nn

from typing import Tuple
from tqdm import tqdm

from planet.planning.action import ActionPlanner


@torch.no_grad()
def latent_planning(
    H: int,
    I: int,
    J: int,
    K: int,
    hidden_state: torch.Tensor,
    current_state_belief: torch.distributions.Normal,
    deterministic_state_model: nn.Module,
    stochastic_state_model: nn.Module,
    reward_model: nn.Module,
    action_size: int,
    action_min: float = -1.0,
    action_max: float = 1.0,
) -> torch.Tensor:
    """
    Implement the latent planning algorithm.

    :param H: Planning horizon distance.
    :param I: Optimization iterations.
    :param J: Candidates per iteration.
    :param K: Top-K candidates to keep.
    :param hidden_state: The hidden state.
    :param current_state_belief: Tuple containing the mean and log std of the current state belief.
    :param deterministic_state_model: The deterministic state model.
    :param stochastic_state_model: The transition model.
    :param reward_model: The reward model.
    :param action_size: The size of the action space.
    :param action_min: The minimum value of the action.
    :param action_max: The maximum value of the action.
    :return: The first action
    """
    action_seq = ActionPlanner(H=H, action_size=action_size)
    hidden_state = hidden_state.repeat(J, 1)

    for _ in range(I):
        reward_sum = torch.zeros((J,)).cuda()

        # sample candidate action sequence
        # (J, H, action_size)
        candidate_actions = action_seq.sample(J)
        candidate_actions = torch.clamp(
            candidate_actions.cuda(), min=action_min, max=action_max
        )

        # initialize the state
        state = current_state_belief.sample((J,)).reshape(J, -1)
        hidden_state_i = hidden_state

        for t in range(H):
            hidden_state_i = deterministic_state_model(
                hidden_state=hidden_state_i,
                state=state,
                action=candidate_actions[:, t],
            )

            # sample the next state
            # (J, state_size)
            prior_dist = stochastic_state_model(hidden_state=hidden_state_i)
            state = prior_dist.sample()

            reward_sum += reward_model(
                hidden_state=hidden_state_i, state=state
            ).reshape(J)

        # select the top-K candidates
        # (K, H, action_size)
        top_k_indices = torch.argsort(reward_sum, descending=True)[:K].tolist()
        top_k_candidates = candidate_actions[top_k_indices]

        # compute mean and std of the top-K candidates
        mean = top_k_candidates.mean(dim=0)
        std = (top_k_candidates - mean.unsqueeze(0)).abs().sum(dim=0) / (K - 1)

        # update the action sequence
        action_seq.update(mean=mean, std=std)

    # return the first action mean
    return action_seq.mean[0]
