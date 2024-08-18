import gym
import torch

from typing import List

from planet.dataset.env_objects import EnvStep
from planet.dataset.buffer import ReplayBuffer


def sample_random_sequences(
    env: gym.Env,
    num_sequences: int = 50,
    max_sequence_len: int = 1000,
) -> List[List[EnvStep]]:
    """
    Sample random sequences from the environment.

    :param env: The environment to sample from.
    :param num_sequences: The number of sequences to sample.
    :param max_sequence_len: The maximum length of each sequence.
    :return: A list of sequences, where each sequence is a list of EnvStep objects.
    """
    sequences = []

    for _ in range(num_sequences):
        sequence = []
        observation, _ = env.reset()

        for _ in range(max_sequence_len):
            action = env.action_space.sample()
            new_observation, reward, terminated, truncated, _ = env.step(
                action
            )

            # add step to the sequence
            done = 1 if terminated or truncated else 0
            sequence.append(
                EnvStep(
                    observation=torch.from_numpy(observation).float(),
                    action=torch.from_numpy(action).float(),
                    reward=reward,
                    done=done,
                )
            )

            if done == 1:
                break

            observation = new_observation

        # add the sequence to the list of sequences
        sequences.append(sequence)

    return sequences


def init_buffer(
    buffer,
    env,
    num_sequences=50,
    max_sequence_len=1000,
) -> ReplayBuffer:

    # sample random sequences from the environment
    sequences = sample_random_sequences(
        env=env, num_sequences=num_sequences, max_sequence_len=max_sequence_len
    )

    # add the sequences to the buffer
    for sequence in sequences:
        buffer.add_sequence(sequence)

    return buffer
