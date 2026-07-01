from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import N_AMINO_ACIDS, N_CODONS, target_profile_position_features


def build_codon_head(in_dim: int, out_dim: int, *, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class CodonProfileNet(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int = 32,
        hidden_dim: int = 64,
        profile_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        window_size: int = 10,
        mm_radius: int = 0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.mm_radius = mm_radius
        self.aa_emb = nn.Embedding(N_AMINO_ACIDS, emb_dim)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.target_proj = nn.Linear(2 * mm_radius + 1, profile_dim)
        self.head = build_codon_head(hidden_dim * 2 + profile_dim, N_CODONS, hidden_dim=hidden_dim)

    def forward(
        self,
        aa: torch.Tensor,
        lengths: torch.Tensor,
        target_profiles: torch.Tensor,
    ) -> torch.Tensor:
        x = self.aa_emb(aa)
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed,
            batch_first=True,
            total_length=aa.size(1),
        )
        target_pos = target_profile_position_features(
            target_profiles,
            lengths,
            aa.size(1),
            window_size=self.window_size,
            mm_radius=self.mm_radius,
        )
        target_feat = self.target_proj(target_pos)
        return self.head(torch.cat([out, target_feat], dim=-1))

    def masked_probs(
        self,
        aa: torch.Tensor,
        lengths: torch.Tensor,
        target_profiles: torch.Tensor,
        syn_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(aa, lengths, target_profiles)
        allowed = syn_mask[aa]
        masked_logits = logits.masked_fill(~allowed, -1e9)
        return F.softmax(masked_logits, dim=-1)
