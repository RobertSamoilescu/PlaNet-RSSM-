import torch.nn as nn
from typing import Any, Dict

from planet.models.reward import RewardModel
from planet.models.encoder import EncoderModel, ImageEncoderModel
from planet.models.observation import ObservationModel, ImageObservationModel
from planet.models.stochastic_state import StochasticStateModel
from planet.models.determinstic_state import DeterministicStateModel


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
