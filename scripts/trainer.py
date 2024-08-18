import yaml  # type: ignore[import-untyped]
import argparse
import numpy as np

from planet.trainer import PlanetTrainer
from planet.utils.seed import set_seed
from planet.utils.envs import make_env
from planet.models.utils import get_models


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

    # set seed for reproducibility and get models
    set_seed(args.seed)
    models = get_models(config)

    trainer = PlanetTrainer(models=models, config=config)
    trainer.fit()
