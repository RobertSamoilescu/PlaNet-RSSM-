import torch
from typing import List, Optional
from planet.dataset.env_objects import EnvStep, EnvSequence


class SequenceBuffer:
    def __init__(self, max_len: int = 1_000) -> None:
        self.max_len = max_len
        self.len = 0

        self.buffer: List[Optional[List[EnvStep]]] = [None] * max_len
        self.idx = 0

    def add_sequence(self, sequence: List[EnvStep]) -> None:
        if self.len < self.max_len:
            self.len += 1

        self.buffer[self.idx] = sequence
        self.idx = (self.idx + 1) % self.max_len

    def _pad_sequence(self, sequence: List[EnvStep], L: int) -> List[EnvStep]:
        if len(sequence) == L:
            return sequence

        pad_step = EnvStep(
            observation=sequence[-1].observation.zero_(),
            action=sequence[-1].action.zero_(),
            reward=0.0,
            done=True,
        )
        return sequence + [pad_step] * (L - len(sequence))

    def _create_sequence(self, sequence: List[EnvStep]) -> EnvSequence:
        observations = torch.stack([step.observation for step in sequence])
        actions = torch.stack([step.action for step in sequence])
        rewards = torch.tensor([step.reward for step in sequence])
        dones = torch.tensor([step.done for step in sequence])
        return EnvSequence(observations, actions, rewards, dones)

    def sample_sequence(self, L: int) -> EnvSequence:
        idx = torch.randint(0, self.len, (1,)).item()
        sequence = self.buffer[idx]  # type: ignore[index]

        start_idx = torch.randint(0, len(sequence), (1,)).item()  # type: ignore[arg-type]  # noqa: E501
        end_idx = min(start_idx + L, len(sequence))  # type: ignore[arg-type]

        sequence = sequence[start_idx:end_idx]  # type: ignore[index, misc]
        sequence = self._pad_sequence(sequence, L)
        return self._create_sequence(sequence)

    def sample_batch(self, B: int, L: int) -> EnvSequence:
        sequences = [self.sample_sequence(L) for _ in range(B)]
        observations = torch.stack([seq.observations for seq in sequences])
        actions = torch.stack([seq.actions for seq in sequences])
        rewards = torch.stack([seq.rewards for seq in sequences])
        dones = torch.stack([seq.dones for seq in sequences])
        return EnvSequence(observations, actions, rewards, dones)
