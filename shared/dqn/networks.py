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
        self.state_dim = state_dim
        self.candidate_dim = candidate_dim
        self.net = _mlp(state_dim + candidate_dim, hidden_dim, 1)

    def forward(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        repeated_state = state.expand(candidates.shape[0], -1)
        features = torch.cat([repeated_state, candidates], dim=1)
        return self.net(features).squeeze(-1)

    def forward_batch(
        self, states: list[torch.Tensor], candidates_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[int]]:
        """Single forward pass for a batch of transitions.

        Returns (Q_values, offsets) where offsets[i] is the start index
        of the i-th transition's candidate block in Q_values.
        """
        offsets = [0]
        rows = []
        for s, cands in zip(states, candidates_list):
            if cands.shape[0] == 0:
                offsets.append(offsets[-1])
                continue
            repeated = s.expand(cands.shape[0], -1)
            rows.append(torch.cat([repeated, cands], dim=1))
            offsets.append(offsets[-1] + cands.shape[0])
        if not rows:
            return (
                torch.empty(0, device=states[0].device),
                offsets,
            )
        flat = torch.cat(rows, dim=0)
        return self.net(flat).squeeze(-1), offsets


class DuelingDQNNet(nn.Module):
    def __init__(self, state_dim: int, candidate_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.candidate_dim = candidate_dim
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

    def forward_batch(
        self, states: list[torch.Tensor], candidates_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[int]]:
        """Single forward pass for a batch of transitions."""
        offsets = [0]
        state_rows: list[torch.Tensor] = []
        cand_rows: list[torch.Tensor] = []
        for _s, cands in zip(states, candidates_list):
            nc = cands.shape[0]
            if nc == 0:
                offsets.append(offsets[-1])
                continue
            state_rows.append(_s)
            cand_rows.append(cands)
            offsets.append(offsets[-1] + nc)

        if not state_rows:
            return torch.empty(0, device=states[0].device), offsets

        state_cat = torch.cat(state_rows, dim=0)   # (N_valid, state_dim)
        cand_cat = torch.cat(cand_rows, dim=0)      # (total_C, cand_dim)
        state_emb_all = self.state_encoder(state_cat)      # (N_valid, hidden)
        cand_emb_all = self.candidate_encoder(cand_cat)    # (total_C, hidden)

        # Map each candidate back to its state embedding row
        cand_to_state = torch.repeat_interleave(
            torch.arange(len(state_rows), device=cand_cat.device),
            torch.tensor([c.shape[0] for c in cand_rows], device=cand_cat.device),
        )
        value_all = self.value_head(state_emb_all).squeeze(-1)

        repeated_state_emb = state_emb_all[cand_to_state]
        advantage_all = self.advantage_head(
            torch.cat([repeated_state_emb, cand_emb_all], dim=1)
        ).squeeze(-1)

        # Dueling: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        results: list[torch.Tensor] = []
        c_offset = 0
        for i, cands in enumerate(cand_rows):
            nc = cands.shape[0]
            v = value_all[i]
            adv = advantage_all[c_offset : c_offset + nc]
            results.append(v.expand_as(adv) + adv - adv.mean())
            c_offset += nc
        return torch.cat(results, dim=0), offsets
