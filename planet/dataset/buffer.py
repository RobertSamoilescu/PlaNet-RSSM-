import torch
import random
import numpy as np
from collections import deque
from typing import List, Optional
from planet.dataset.env_objects import EnvStep, EnvSequence


class ReplayBuffer:
    def __init__(self, max_len: int = 1_000) -> None:
        """Replay buffer for storing sequences of environment steps

        :param max_len: Maximum number of sequences to store
        """
        self.buffer = deque(maxlen=max_len)

    def add_sequence(self, sequence: List[EnvStep]) -> None:
        """Add a sequence of environment steps to the buffer

        :param sequence: Sequence to be added
        :return: None
        """
        self.buffer.append(sequence)

    def _pad_sequence(self, sequence: List[EnvStep], L: int) -> List[EnvStep]:
        """Pad a sequence to length L with zero steps

        :param sequence: Sequence to be padded
        :param L: Length of the padded sequence
        :return: Padded sequence
        """
        if len(sequence) == L:
            return sequence

        pad_step = EnvStep(
            observation=torch.zeros_like(sequence[-1].observation),
            action=torch.zeros_like(sequence[-1].action),
            reward=0.0,
            done=1,
        )
        return sequence + [pad_step] * (L - len(sequence))

    def _create_sequence(self, sequence: List[EnvStep]) -> EnvSequence:
        """Create an EnvSequence object from a list of EnvStep objects

        :param sequence: List of EnvStep objects
        :return: EnvSequence object
        """
        observations = torch.stack([step.observation for step in sequence])
        actions = torch.stack([step.action for step in sequence])
        rewards = torch.tensor([step.reward for step in sequence])
        dones = torch.tensor([step.done for step in sequence])
        return EnvSequence(observations, actions, rewards, dones)

    def sample_sequence(self, L: int) -> EnvSequence:
        """Sample a sequence of length L from the buffer

        :param L: Length of the sequence to sample
        :return: Sampled sequence
        """
        sequence = random.sample(self.buffer, 1)[0]
        start_idx = torch.randint(0, max(len(sequence) - L, 0) + 1, (1,)).item()  # type: ignore[arg-type]  # noqa: E501
        end_idx = min(start_idx + L, len(sequence))  # type: ignore[arg-type]

        sequence = sequence[start_idx:end_idx]  # type: ignore[index, misc]
        sequence = self._pad_sequence(sequence, L)
        assert len(sequence) == L
        return self._create_sequence(sequence)

    def sample_batch(self, B: int, L: int) -> EnvSequence:
        """Sample a batch of sequences from the buffer

        :param B: Batch size
        :param L: Length of the sequences to sample
        :return: Batch of sequences
        """
        sequences = [self.sample_sequence(L) for _ in range(B)]
        observations = torch.stack([seq.observations for seq in sequences])
        actions = torch.stack([seq.actions for seq in sequences])
        rewards = torch.stack([seq.rewards for seq in sequences])
        dones = torch.stack([seq.dones for seq in sequences])
        return EnvSequence(observations, actions, rewards, dones)
