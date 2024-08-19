import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from planet.models.reward import RewardModel
from planet.models.encoder import EncoderModel, ImageEncoderModel
from planet.models.observation import ObservationModel, ImageObservationModel
from planet.models.stochastic_state import StochasticStateModel
from planet.models.determinstic_state import DeterministicStateModel


def zero_grad(optimizers: Dict[str, torch.optim.Optimizer]):
    for optimizer in optimizers.values():
        optimizer.zero_grad()


def gradient_step(optimizers: Dict[str, torch.optim.Optimizer]):
    for optimizer in optimizers.values():
        optimizer.step()


def set_models_eval(models: Dict[str, nn.Module]):
    for model in models.values():
        model.eval()


def set_models_train(models: Dict[str, nn.Module]):
    for model in models.values():
        model.train()


def save_models(
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    path: str,
):
    checkpoint = {}

    for key, model in models.items():
        checkpoint[key] = model.state_dict()

    for key, optimizer in optimizers.items():
        checkpoint[key] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_models(
    models: Dict[str, nn.Module],
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    path: str = "checkpoints/best_model.pth",
):
    checkpoint = torch.load(path)

    for key, model in models.items():
        model.load_state_dict(checkpoint[key])

    if optimizers is not None:
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint[key])


def get_models(config: Dict[str, Any]) -> Dict[str, nn.Module]:
    """
    Get models

    :param config: configuration dictionary
    :return: dictionary of models
    """
    model_config = config["model_config"]

    # construct the deterministic state model
    det_state_model = DeterministicStateModel(
        rnn_hidden_size=model_config["rnn_hidden_size"],
        state_size=model_config["state_size"],
        action_size=model_config["action_size"],
        hidden_size=model_config["hidden_size"],
    ).to(model_config["device"])

    # construct the stochastic state model
    stoch_state_model = StochasticStateModel(
        rnn_hidden_size=model_config["rnn_hidden_size"],
        state_size=model_config["state_size"],
        hidden_size=model_config["hidden_size"],
        min_std=model_config["min_std"],
    ).to(model_config["device"])

    # construct the reward observation model
    reward_obs_model = RewardModel(
        rnn_hidden_size=model_config["rnn_hidden_size"],
        state_size=model_config["state_size"],
        hidden_size=model_config["hidden_size"],
    ).to(model_config["device"])

    # construct the observation model
    obs_klass = (
        ImageObservationModel
        if config["env_config"]["env_type"] == "dm_control"
        else ObservationModel
    )
    obs_model = obs_klass(
        rnn_hidden_size=model_config["rnn_hidden_size"],
        state_size=model_config["state_size"],
        observation_size=model_config["observation_size"],
        hidden_size=model_config["hidden_size"],  # type: ignore[call-arg]
    ).to(model_config["device"])

    # construct the encoder model
    enc_klass = (
        ImageEncoderModel
        if config["env_config"]["env_type"] == "dm_control"
        else EncoderModel
    )
    enc_model = enc_klass(
        rnn_hidden_size=model_config["rnn_hidden_size"],
        observation_size=model_config["observation_size"],
        state_size=model_config["state_size"],
        hidden_size=model_config["hidden_size"],
        min_std=model_config["min_std"],
    ).to(model_config["device"])

    return {
        "det_state_model": det_state_model,
        "stoch_state_model": stoch_state_model,
        "obs_model": obs_model,
        "reward_obs_model": reward_obs_model,
        "enc_model": enc_model,
    }
