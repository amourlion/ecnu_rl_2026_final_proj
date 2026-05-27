from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class DQNNet(nn.Module):
    def __init__(self, state_dim: int, candidate_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = _mlp(state_dim + candidate_dim, hidden_dim, 1)

    def forward(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        repeated_state = state.expand(candidates.shape[0], -1)
        features = torch.cat([repeated_state, candidates], dim=1)
        return self.net(features).squeeze(-1)


class DuelingDQNNet(nn.Module):
    def __init__(self, state_dim: int, candidate_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_encoder = _mlp(state_dim, hidden_dim, hidden_dim)
        self.candidate_encoder = _mlp(candidate_dim, hidden_dim, hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        state_emb = self.state_encoder(state)
        candidate_emb = self.candidate_encoder(candidates)
        repeated_state = state_emb.expand(candidates.shape[0], -1)
        value = self.value_head(state_emb).squeeze(-1)
        advantage = self.advantage_head(
            torch.cat([repeated_state, candidate_emb], dim=1)
        ).squeeze(-1)
        return value.expand_as(advantage) + advantage - advantage.mean()
