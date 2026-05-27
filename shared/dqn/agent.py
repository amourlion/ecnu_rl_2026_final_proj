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
                state=encoded.state.detach(),
                candidates=encoded.candidates.detach(),
                action=action,
                reward=reward,
                next_state=encoded_next.state.detach(),
                next_candidates=encoded_next.candidates.detach(),
                done=done,
            )
        )

    def optimize(self, step: int) -> float | None:
        if len(self.replay) < self.config.min_replay_size:
            return None
        batch = self.replay.sample(self.config.batch_size, beta=self.config.beta_at(step))
        losses = []
        td_errors = []
        for transition in batch.transitions:
            q_values = self.policy_net(transition.state, transition.candidates)
            q_value = q_values[transition.action]
            with torch.no_grad():
                if transition.done or transition.next_candidates.shape[0] == 0:
                    next_value = torch.tensor(0.0, device=self.device)
                elif self.config.double_dqn:
                    next_online = self.policy_net(
                        transition.next_state,
                        transition.next_candidates,
                    )
                    next_action = int(torch.argmax(next_online).item())
                    next_value = self.target_net(
                        transition.next_state,
                        transition.next_candidates,
                    )[next_action]
                else:
                    next_value = self.target_net(
                        transition.next_state,
                        transition.next_candidates,
                    ).max()
                target = torch.tensor(transition.reward, device=self.device) + (
                    self.config.gamma * next_value
                )
            loss = self.loss_fn(q_value, target)
            losses.append(loss)
            td_errors.append(q_value.detach() - target.detach())

        weights = batch.weights.to(self.device)
        loss_tensor = torch.stack(losses)
        loss = (loss_tensor * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.replay.update_priorities(batch.indices, torch.stack(td_errors))
        return float(loss.detach().cpu().item())

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _select_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
