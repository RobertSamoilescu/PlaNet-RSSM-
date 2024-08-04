import torch
import random
from collections import deque
from typing import List, Optional
from planet.dataset.env_objects import EnvStep, EnvSequence


class SequenceBuffer:
    def __init__(self, max_len: int = 1_000) -> None:
        self.buffer = deque(maxlen=max_len)

    def add_sequence(self, sequence: List[EnvStep]) -> None:
        self.buffer.append(sequence)

    def _pad_sequence(self, sequence: List[EnvStep], L: int) -> List[EnvStep]:
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
        observations = torch.stack([step.observation for step in sequence])
        actions = torch.stack([step.action for step in sequence])
        rewards = torch.tensor([step.reward for step in sequence])
        dones = torch.tensor([step.done for step in sequence])
        return EnvSequence(observations, actions, rewards, dones)

    def sample_sequence(self, L: int) -> EnvSequence:
        sequence = random.sample(self.buffer, 1)[0]
        start_idx = torch.randint(0, len(sequence), (1,)).item()  # type: ignore[arg-type]  # noqa: E501
        end_idx = min(start_idx + L, len(sequence))  # type: ignore[arg-type]

        sequence = sequence[start_idx:end_idx]  # type: ignore[index, misc]
        sequence = self._pad_sequence(sequence, L)
        assert len(sequence) == L
        return self._create_sequence(sequence)

    def sample_batch(self, B: int, L: int) -> EnvSequence:
        sequences = [self.sample_sequence(L) for _ in range(B)]
        observations = torch.stack(
            [seq.observations for seq in sequences]
        ).float()
        actions = torch.stack([seq.actions for seq in sequences]).float()
        rewards = torch.stack([seq.rewards for seq in sequences]).float()
        dones = torch.stack([seq.dones for seq in sequences]).float()
        return EnvSequence(observations, actions, rewards, dones)
