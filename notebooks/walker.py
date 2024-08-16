# %%
import torch.optim as optim
from planet.models.determinstic_state import DeterministicStateModel
from planet.models.stochastic_state import StochasticStateModel
from planet.models.reward import RewardModel
from planet.models.encoder import ImageEncoderModel
from planet.models.observation import ImageObservationModel
from planet.trainer import PlanetTrainer

from planet.utils.wrappers import RepeatActionWrapper, GymPixelWrapper, ImagePreprocessorWrapper
from planet.utils.seed import set_seed
from planet.utils.envs import make_env



# %%
# set seed for reproducibility
set_seed(13)

# %%
free_nats = 2.0
action_repeat = 2
env_config = {
    "env_type": "dm_control",
    "domain_name": "walker", 
    "task_name":"walk", 
    "render_kwargs": {'width': 64, 'height': 64, 'camera_id': 0},
    "skip": action_repeat
}

env = make_env(env_config)

# %%
# action info
action = env.action_space.sample()
action_size = action.shape[0]

# model sizes
state_size = 30
hidden_state_size = 200
observation_size = 1024
hidden_layer_size = 200

# %%
det_state_model = DeterministicStateModel(
    hidden_state_size=hidden_state_size,
    state_size=state_size,
    action_size=action_size,
    hidden_layer_size=hidden_layer_size
).cuda()

# %%
stoch_state_model = StochasticStateModel(
    hidden_state_size=hidden_state_size,
    state_size=state_size,
    hidden_layer_size=hidden_layer_size,
).cuda()

# %%
obs_model = ImageObservationModel(
    hidden_state_size=hidden_state_size,
    state_size=state_size,
    observation_size=observation_size,
).cuda()

# %%
reward_obs_model = RewardModel(
    hidden_state_size=hidden_state_size,
    state_size=state_size,
    hidden_layer_size=hidden_layer_size,
).cuda()

# %%
enc_model = ImageEncoderModel(
    hidden_state_size=hidden_state_size,
    observation_size=observation_size,
    state_size=state_size,
    hidden_layer_size=hidden_layer_size,
).cuda()

# %%
models = {
    "det_state_model": det_state_model,
    "stoch_state_model": stoch_state_model,
    "obs_model": obs_model,
    "reward_obs_model": reward_obs_model,
    "enc_model": enc_model,
}


lr = 6e-4
all_params = list(det_state_model.parameters()) + list(stoch_state_model.parameters()) + list(obs_model.parameters()) + list(reward_obs_model.parameters()) + list(enc_model.parameters())

optimizers = {
    "all_params": optim.Adam(
        all_params,
        lr=lr, 
    ),
}



# %%
trainer = PlanetTrainer(
    models=models,
    optimizers=optimizers,
    config={
        "env_config": env_config,
        "train_config": {
            "S": 5,
            "train_steps": 2_000,
            "C": 100,
            "B": 50,
            "L": 50,
            "H": 15,
            "I": 10,
            "J": 1000,
            "K": 100,
            "log_interval": 1,
            "action_noise": 0.1,
            "free_nats": free_nats,
            "checkpoint_dir": "checkpoints-walker-%.2f" % free_nats,
            "max_episode_length": 1000,
            "action_repeat": action_repeat,
            "all_params": all_params
        },
        "state_config": {
            "hidden_state_size": hidden_state_size,
            "state_size": state_size,
            "action_size": action_size,
        },
        "eval_config": {
            "eval_interval": 25,
            "num_eval_episodes": 5,
        }
    }
)

# %%
trainer.fit()

# %%



