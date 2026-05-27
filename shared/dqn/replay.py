from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Transition:
    state: torch.Tensor
    candidates: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    next_candidates: torch.Tensor
    done: bool


@dataclass(slots=True)
class ReplayBatch:
    transitions: list[Transition]
    indices: list[int]
    weights: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: list[Transition] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 1.0) -> ReplayBatch:
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        transitions = [self.buffer[index] for index in indices]
        weights = torch.ones(len(transitions), dtype=torch.float32)
        return ReplayBatch(transitions=transitions, indices=indices, weights=weights)

    def update_priorities(self, indices: list[int], td_errors: torch.Tensor) -> None:
        return None


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-6,
    ) -> None:
        super().__init__(capacity)
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.priorities: list[float] = []

    def push(self, transition: Transition) -> None:
        max_priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> ReplayBatch:
        sample_size = min(batch_size, len(self.buffer))
        scaled = torch.tensor(self.priorities[: len(self.buffer)], dtype=torch.float32)
        scaled = scaled.pow(self.alpha)
        probabilities = scaled / scaled.sum()
        indices_tensor = torch.multinomial(probabilities, sample_size, replacement=False)
        indices = [int(index) for index in indices_tensor.tolist()]
        transitions = [self.buffer[index] for index in indices]
        weights = (len(self.buffer) * probabilities[indices_tensor]).pow(-beta)
        weights = weights / weights.max()
        return ReplayBatch(
            transitions=transitions,
            indices=indices,
            weights=weights.detach().float(),
        )

    def update_priorities(self, indices: list[int], td_errors: torch.Tensor) -> None:
        for index, td_error in zip(indices, td_errors.detach().abs().cpu().tolist()):
            self.priorities[index] = float(td_error + self.priority_epsilon)
