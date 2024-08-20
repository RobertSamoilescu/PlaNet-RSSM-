import os
import yaml  # type: ignore[import-untyped]
import argparse
import numpy as np

from planet.utils.vis import images_to_gif
from planet.utils.envs import make_env
from planet.models.utils import get_models, load_models, set_models_eval
from planet.collector import collect_episode


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Planet Pytorch")
    parser.add_argument(
        "--seed", type=int, default=13, help="Seed for reproducibility"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dm_control/walker_walk.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to test",
    )
    parser.add_argument(
        "-J",
        type=int,
        default=1000,
        help="Number of action candidates to sample"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # read yaml config file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # add configs in not set
    env = make_env(config["env_config"])
    if "action_size" not in config["model_config"]:
        config["model_config"]["action_size"] = np.prod(env.action_space.shape)

    if "observation_size" not in config["model_config"]:
        config["model_config"]["observation_size"] = np.prod(
            env.observation_space.shape
        )

    # add J to config
    config["train_config"]["J"] = args.J  

    # load models
    path = os.path.join(
        config["train_config"]["checkpoint_dir"], "best_model.pth"
    )
    models = get_models(config)
    load_models(models, path=path)
    set_models_eval(models)

    # compute max episode length
    T = config["train_config"]["T"]
    skip = config["env_config"]["skip"]
    max_episode_length = T // skip

    # test the models
    rewards, observations = [], []
    for _ in range(args.num_episodes):
        episode = collect_episode(
            env=env,
            models=models,
            rnn_hidden_size=config["model_config"]["rnn_hidden_size"],
            action_size=config["model_config"]["action_size"],
            max_episode_length=max_episode_length,
            H=config["train_config"]["H"],
            I=config["train_config"]["I"],
            J=config["train_config"]["J"],
            K=config["train_config"]["K"],
            action_min=config["train_config"]["action_min"],
            action_max=config["train_config"]["action_max"],
            action_noise=None,
            device=config["model_config"]["device"],
            return_obs=True,
        )

        rewards.append(episode["reward"])
        observations.append(episode["observations"])

    # print statistics
    print(
        "Mean Reward: %.4f, Median Reward: %.4f, "
        "Std Reward: %.4f, Min Reward: %.4f, Max Reward: %.4f"
        % (
            np.mean(rewards),
            np.median(rewards),
            np.std(rewards),
            np.min(rewards),
            np.max(rewards),
        )
    )

    # save best episode observations
    best_idx = np.argmax(rewards)
    best_obs = observations[best_idx]
    path = os.path.join(
        config["train_config"]["checkpoint_dir"], "best_episode.gif"
    )
    images_to_gif(best_obs, path)
