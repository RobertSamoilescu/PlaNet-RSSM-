import torch
import torch.nn as nn

from typing import Tuple
from tqdm import tqdm

from planet.planning.action import ActionPlanner


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

    for i in range(I):
        rewards = []
        candidates_actions = []

        for j in range(J):
            reward_sum = 0

            # sample candidate action sequence
            # (H, action_size)
            candidate_actions = action_seq.sample()

            # initialize state
            mean_state, log_std_state = current_state_belief
            state = torch.distributions.Normal(mean_state, log_std_state.exp()).sample()

            for t in range(H):
                # compute the next hidden state
                hidden_state = deterministic_state_model(
                    hidden_state=hidden_state,
                    state=state.reshape(1, 1, -1),
                    action=candidate_actions[t].reshape(1, 1, -1)
                )
                
                # sample the next state
                mean_state, log_std_state = stochastic_state_model(hidden_state=hidden_state.reshape(1, -1))
                state = torch.distributions.Normal(mean_state, log_std_state.exp()).sample()

                # compute the reward
                reward_sum += reward_model(hidden_state=hidden_state.reshape(1, -1), state=state)
        
            rewards.append(reward_sum.item())
            candidates_actions.append(candidate_actions)

        # stack the candidate actions
        # (J, H, action_size)
        candidates_actions = torch.stack(candidates_actions)
        rewards = torch.tensor(rewards)

        # select the top-K candidates
        # (K, H, action_size)
        top_k_indices = torch.argsort(rewards, descending=True)[:K]
        top_k_candidates = candidates_actions[top_k_indices]

        # compute mean and std of the top-K candidates
        mean = top_k_candidates.mean(dim=0)
        std = top_k_candidates.std(dim=0)

        # update the action sequence
        action_seq.update(mean=mean, std=std)

    # return the first action mean
    return action_seq.actions[0].mean



