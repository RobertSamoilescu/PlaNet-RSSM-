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
    current_state_belief: Tuple[torch.Tensor, torch.Tensor],
    deterministic_state_model: nn.Module,
    stochastic_state_model: nn.Module,
    reward_model: nn.Module,
    action_size: int,
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
    :return: The first action
    """
    action_seq = ActionPlanner(H=H, action_size=action_size)
    hidden_state = torch.tile(hidden_state, (J, 1))

    for i in range(I):
        reward_sum = torch.zeros((J, )).cuda()

        # sample candidate action sequence
        # (J, H, action_size)
        candidate_actions = action_seq.sample(J)
        candidate_actions = candidate_actions.cuda()

        # initialize state
        # (1, state_size), (1, state_size)
        # (J, state_size)
        state = current_state_belief
        state = torch.tile(state, (J, 1))

        # save hidden state for the next iteration
        hidden_state_i = hidden_state

        for t in range(H):
            # compute the next hidden state
            hidden_state_i = deterministic_state_model(
                hidden_state=hidden_state_i,
                state=state.reshape(J, 1, -1),
                action=candidate_actions[:, t].reshape(J, 1, -1),
            )

            # sample the next state
            # (J, state_size)
            # (J, state_size)
            mean_state, log_std_state = stochastic_state_model(
                hidden_state=hidden_state_i.reshape(J, -1)
            )
            state = torch.distributions.Normal(
                mean_state, log_std_state.exp()
            ).sample()

            # compute the reward
            reward_sum += reward_model(
                hidden_state=hidden_state_i.reshape(J, -1), 
                state=state
            ).reshape(J)


        # select the top-K candidates
        # (K, H, action_size)
        top_k_indices = torch.argsort(reward_sum, descending=True)[:K].tolist()
        top_k_candidates = candidate_actions[top_k_indices]

        # compute mean and std of the top-K candidates
        mean = top_k_candidates.mean(dim=0)
        std = torch.clip(top_k_candidates.std(dim=0), 0.01)

        # update the action sequence
        action_seq.update(mean=mean, std=std)

    # return the first action mean
    return action_seq.mean[0]
