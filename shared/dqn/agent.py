from __future__ import annotations

import random

import torch
from torch import nn

from shared.dqn.config import DQNConfig
from shared.dqn.features import FeatureEncoder
from shared.dqn.networks import DQNNet, DuelingDQNNet
from shared.dqn.replay import PrioritizedReplayBuffer, ReplayBuffer, Transition


class DQNAgent:
    def __init__(self, config: DQNConfig) -> None:
        self.config = config
        self.device = self._select_device(config.device)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.encoder = FeatureEncoder(self.device)
        network_cls = DuelingDQNNet if config.network_type == "dueling" else DQNNet
        self.policy_net = network_cls(
            self.encoder.state_dim,
            self.encoder.candidate_dim,
            config.hidden_dim,
        ).to(self.device)
        self.target_net = network_cls(
            self.encoder.state_dim,
            self.encoder.candidate_dim,
            config.hidden_dim,
        ).to(self.device)
        self.sync_target()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        if config.replay_type == "prioritized":
            self.replay = PrioritizedReplayBuffer(
                config.replay_capacity,
                alpha=config.per_alpha,
                priority_epsilon=config.priority_epsilon,
            )
        else:
            self.replay = ReplayBuffer(config.replay_capacity)

    def act(self, state: dict, candidates, epsilon: float = 0.0) -> int:
        if len(candidates) == 0:
            raise ValueError("cannot act without candidates")
        if random.random() < epsilon:
            return random.randrange(len(candidates))
        with torch.no_grad():
            encoded = self.encoder.encode(state, candidates)
            q_values = self.policy_net(encoded.state, encoded.candidates)
        return int(torch.argmax(q_values).item())

    def push_transition(
        self,
        state: dict,
        candidates,
        action: int,
        reward: float,
        next_state: dict,
        next_candidates,
        done: bool,
    ) -> None:
        encoded = self.encoder.encode(state, candidates)
        encoded_next = self.encoder.encode(next_state, next_candidates)
        self.replay.push(
            Transition(
                state=encoded.state.detach().cpu(),
                candidates=encoded.candidates.detach().cpu(),
                action=action,
                reward=reward,
                next_state=encoded_next.state.detach().cpu(),
                next_candidates=encoded_next.candidates.detach().cpu(),
                done=done,
            )
        )

    def optimize(self, step: int) -> float | None:
        if len(self.replay) < self.config.min_replay_size:
            return None
        batch = self.replay.sample(self.config.batch_size, beta=self.config.beta_at(step))

        batch_Q, _action_offsets = self.policy_net.forward_batch(
            [t.state.to(self.device) for t in batch.transitions],
            [t.candidates.to(self.device) for t in batch.transitions],
        )

        # Index into the concatenated Q-values at each action location
        q_values = []
        for i, t in enumerate(batch.transitions):
            start = _action_offsets[i]
            q_values.append(batch_Q[start + t.action])
        q_value = torch.stack(q_values)

        with torch.no_grad():
            next_batch_Q, _next_offsets = self.target_net.forward_batch(
                [t.next_state.to(self.device) for t in batch.transitions],
                [t.next_candidates.to(self.device) for t in batch.transitions],
            )
            if self.config.double_dqn:
                next_online_batch_Q, _next_online_offsets = self.policy_net.forward_batch(
                    [t.next_state.to(self.device) for t in batch.transitions],
                    [t.next_candidates.to(self.device) for t in batch.transitions],
                )
            next_values = []
            for i, t in enumerate(batch.transitions):
                start = _next_offsets[i]
                end = _next_offsets[i + 1] if i + 1 < len(_next_offsets) else next_batch_Q.shape[0]
                n_candidates = end - start
                if t.done or n_candidates == 0:
                    next_values.append(torch.tensor(0.0, device=self.device))
                elif self.config.double_dqn:
                    q_vals = next_batch_Q[start:end]
                    online_start = _next_online_offsets[i]
                    online_end = (
                        _next_online_offsets[i + 1]
                        if i + 1 < len(_next_online_offsets)
                        else next_online_batch_Q.shape[0]
                    )
                    next_online_q = next_online_batch_Q[online_start:online_end]
                    best = int(torch.argmax(next_online_q).item())
                    next_values.append(q_vals[best])
                else:
                    next_values.append(next_batch_Q[start:end].max())

        target = torch.tensor(
            [t.reward for t in batch.transitions],
            dtype=torch.float32,
            device=self.device,
        ) + self.config.gamma * torch.stack(next_values)

        td_errors = q_value - target

        weights = batch.weights.to(self.device)
        loss = (self.loss_fn(q_value, target) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.replay.update_priorities(batch.indices, td_errors)
        return float(loss.detach().cpu().item())

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _select_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_built()
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
