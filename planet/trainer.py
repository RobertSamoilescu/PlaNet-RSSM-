import torch
import torch.nn as nn

import os
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any

from planet.dataset.buffer import ReplayBuffer
from planet.utils.sample import init_buffer
from planet.utils.envs import make_env
from planet.collector import collect_episode
from planet.loss import (
    compute_observation_loss,
    compute_reward_loss,
    compute_kl_divergence,
)
from planet.models.utils import (
    zero_grad,
    gradient_step,
    set_models_eval,
    set_models_train,
    save_models,
)


class PlanetTrainer:
    def __init__(
        self,
        models: Dict[str, nn.Module],
        config: Dict[str, Any],
    ) -> None:
        """Initialize the trainer with models and config.

        :param models: A dictionary of models
        :param config: A dictionary of configuration parameters
        """
        self.config = config
        self.models = models

        # get all parameters
        self.all_params = []
        for model in models.values():
            self.all_params += list(model.parameters())

        # initialize optimizers
        self.optimizers: Dict[str, torch.optim.Optimizer] = {
            "all_models": torch.optim.Adam(
                self.all_params, lr=self.config["train_config"]["lr"]
            )
        }

        # create checkpoint directory
        self.best_checkpoint_path = os.path.join(
            self.config["train_config"]["checkpoint_dir"], "best_model.pth"
        )
        self.latest_checkpoint_path = os.path.join(
            self.config["train_config"]["checkpoint_dir"], "latest_model.pth"
        )
        if not os.path.exists(self.config["train_config"]["checkpoint_dir"]):
            os.makedirs(self.config["train_config"]["checkpoint_dir"])

        # create environment
        self.env = make_env(self.config["env_config"])

        # create replay buffer
        self.buffer = ReplayBuffer()

        # compute max episode length
        T = self.config["train_config"]["T"]
        skip = self.config["env_config"]["skip"]
        self.max_episode_len = T // skip

    def model_fit_step(self) -> Dict[str, float]:
        # define loss variables
        device = self.config["model_config"].get("device", "cuda")
        obs_loss = torch.tensor(0.0, device=device)
        reward_loss = torch.tensor(0.0, device=device)
        kl_div = torch.tensor(0.0, device=device)

        # sample a batch of experiences
        batch = self.buffer.sample_batch(
            B=self.config["train_config"]["B"],
            L=self.config["train_config"]["L"],
        ).to(device)

        # initialize first hidden state
        hidden_state = torch.zeros(
            self.config["train_config"]["B"],
            self.config["model_config"]["rnn_hidden_size"],
        ).to(device)

        # initialize posterior distribution
        posterior_dist = self.models["enc_model"](
            observation=batch.observations[:, 0], hidden_state=hidden_state
        )

        # sample initial state
        posterior = posterior_dist.rsample()

        for t in range(1, self.config["train_config"]["L"]):
            # compute deterministic hidden state
            next_hidden_state = self.models["det_state_model"](
                hidden_state=hidden_state,
                state=posterior,
                action=batch.actions[:, t - 1],
            )

            # compute the prior distribution of the next state
            next_prior_dist = self.models["stoch_state_model"](
                next_hidden_state
            )

            # compute next approximation of the state
            next_posterior_dist = self.models["enc_model"](
                hidden_state=next_hidden_state,
                observation=batch.observations[:, t],
            )

            # compute the kl divergence between the posterior
            mask = 1 - batch.dones[:, t - 1]
            kl_div += compute_kl_divergence(
                posterior=next_posterior_dist,
                prior_dist=next_prior_dist,
                mask=mask,
                free_nats=self.config["train_config"]["free_nats"],
            )

            # sample next state
            next_posterior = next_posterior_dist.rsample()

            # compute next observation based on the
            # hidden state and the next state
            obs = self.models["obs_model"](
                hidden_state=next_hidden_state,
                state=next_posterior,
            )

            # compute next reward based on the hidden state
            # and the next state
            reward = self.models["reward_obs_model"](
                hidden_state=next_hidden_state,
                state=next_posterior,
            ).reshape(-1)

            # comput observation loss
            gt_obs = batch.observations[:, t]
            obs_loss += compute_observation_loss(gt_obs, obs, mask)

            # compute reward loss
            gt_reward = batch.rewards[:, t - 1]
            reward_loss += compute_reward_loss(gt_reward, reward, mask)

            # update hidden state
            hidden_state = next_hidden_state
            posterior = next_posterior

        # zero gradients
        zero_grad(self.optimizers)

        # compute average loss
        mask_sum = (1 - batch.dones[:, :-1]).sum()
        obs_loss /= mask_sum
        reward_loss /= mask_sum
        kl_div /= mask_sum

        # compute loss
        loss = obs_loss + reward_loss + kl_div
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.all_params, self.config["train_config"]["grad_norm_clip"]
        )

        # gradient step
        gradient_step(self.optimizers)
        return {
            "loss": loss.item(),
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl_div": kl_div.item(),
        }

    def update_running_stats(
        self, stats: Dict[str, float], running_stats: Dict[str, float]
    ) -> Dict[str, float]:
        """Update running statistics.

        :param stats: A dictionary of stats values
        :param running_stats: A dictionary of running statistics
        :return: A dictionary of updated running statistics
        """
        for key, value in stats.items():
            running_stats[key] = (
                value
                if running_stats.get(key) is None
                else 0.99 * running_stats[key] + 0.01 * value
            )
        return running_stats

    def data_collection(self) -> Dict[str, float]:
        """Data collection step."""
        episode = collect_episode(
            env=self.env,
            models=self.models,
            rnn_hidden_size=self.config["model_config"]["rnn_hidden_size"],
            action_size=self.config["model_config"]["action_size"],
            max_episode_length=self.max_episode_len,
            H=self.config["train_config"]["H"],
            I=self.config["train_config"]["I"],
            J=self.config["train_config"]["J"],
            K=self.config["train_config"]["K"],
            action_min=self.config["train_config"]["action_min"],
            action_max=self.config["train_config"]["action_max"],
            action_noise=self.config["train_config"]["action_noise"],
        )
        self.buffer.add_sequence(episode["sequence"])
        return {"reward": episode["reward"]}

    def train_step(self, i: int, running_stats: Dict[str, float] = {}):
        set_models_train(self.models)

        # fit the world model
        for _ in tqdm(
            range(self.config["train_config"]["C"]), desc="Model fit"
        ):
            stats = self.model_fit_step()
            running_stats = self.update_running_stats(stats, running_stats)

        # data collection
        stats = self.data_collection()
        running_stats = self.update_running_stats(stats, running_stats)

        if i % self.config["train_config"]["log_interval"] == 0:
            print(
                "Iter: %d, "
                "Loss: %.4f, Obs Loss: %.4f, Reward Loss: %.4f, KL Div: %.4f, "
                "Collection reward: %.4f"
                % (
                    i,
                    running_stats["loss"],
                    running_stats["obs_loss"],
                    running_stats["reward_loss"],
                    running_stats["kl_div"],
                    running_stats["reward"],
                )
            )

    def _evaluate(self) -> List[float]:
        set_models_eval(self.models)
        episode_rewards = []

        num_eval_episodes = self.config["eval_config"]["num_eval_episodes"]
        for _ in tqdm(range(num_eval_episodes), desc="Evaluation"):
            episode_rewards.append(
                collect_episode(
                    env=self.env,
                    models=self.models,
                    rnn_hidden_size=self.config["model_config"][
                        "rnn_hidden_size"
                    ],
                    action_size=self.config["model_config"]["action_size"],
                    max_episode_length=self.max_episode_len,
                    H=self.config["train_config"]["H"],
                    I=self.config["train_config"]["I"],
                    J=self.config["train_config"]["J"],
                    K=self.config["train_config"]["K"],
                    action_min=self.config["train_config"]["action_min"],
                    action_max=self.config["train_config"]["action_max"],
                    action_noise=None,
                )[1]
            )

        return episode_rewards

    def evaluate_step(self, i: int, best_score: Dict[str, float] = {}) -> None:
        eval_interval = self.config["eval_config"]["eval_interval"]
        train_steps = self.config["train_config"]["train_steps"]

        if "mean_reward" not in best_score:
            best_score["mean_reward"] = -np.inf

        if i % eval_interval == 0 or i == train_steps - 1:
            rewards = self._evaluate()
            mean_reward = np.mean(rewards).item()

            if mean_reward > best_score["mean_reward"]:
                # update best score and save model
                best_score["mean_reward"] = mean_reward
                save_models(
                    models=self.models,
                    optimizers=self.optimizers,
                    path=self.best_checkpoint_path,
                )

            # save latest model
            save_models(
                models=self.models,
                optimizers=self.optimizers,
                path=self.latest_checkpoint_path,
            )

            print(
                "Iter: %d, Mean Reward: %.4f, Median Reward: %.4f, "
                "Std Reward: %.4f, Min Reward: %.4f, Max Reward: %.4f"
                % (
                    i,
                    mean_reward,
                    np.median(rewards),
                    np.std(rewards),
                    np.min(rewards),
                    np.max(rewards),
                )
            )

    def fit(self):
        train_steps = self.config["train_config"]["train_steps"]

        # initialize buffer with S random seeds episodes
        self.buffer = init_buffer(
            buffer=self.buffer,
            env=self.env,
            num_sequences=self.config["train_config"]["S"],
            max_sequence_len=self.max_episode_len,
        )

        for i in range(train_steps):
            self.train_step(i)
            self.evaluate_step(i)
