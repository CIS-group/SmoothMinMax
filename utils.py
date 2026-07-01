from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

NUC_ORDER = "ACGT"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
N_AMINO_ACIDS = 20
N_CODONS = 64

GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def codon_to_index(codon: str) -> int:
    idx = 0
    for ch in codon.upper():
        idx = idx * 4 + NUC_ORDER.index(ch)
    return idx


def aa_to_index(aa: str) -> int:
    return AMINO_ACIDS.index(aa)


def encode_aa_sequence(aa_sequence: str) -> torch.Tensor:
    return torch.tensor([aa_to_index(aa) for aa in aa_sequence], dtype=torch.long)


def build_synonymous_codons() -> dict[str, list[str]]:
    synonymous: dict[str, list[str]] = {}
    for codon, aa in GENETIC_CODE.items():
        if aa == "*":
            continue
        synonymous.setdefault(aa, []).append(codon)
    for codons in synonymous.values():
        codons.sort()
    return synonymous


def build_synonymous_mask() -> torch.Tensor:
    mask = torch.zeros(N_AMINO_ACIDS, N_CODONS, dtype=torch.bool)
    for codon, aa in GENETIC_CODE.items():
        if aa == "*":
            continue
        mask[aa_to_index(aa), codon_to_index(codon)] = True
    return mask


def parse_codon_usage_csv(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    pairs = re.findall(r"([ACGT]{3})\s*,\s*\??([0-9]+(?:\.[0-9]+)?)", text.upper())
    usage = {codon: float(value) for codon, value in pairs}
    if len(usage) < 61:
        raise ValueError(f"Could not parse enough codons from usage table: {path}")
    return usage


def parse_sequences(path: Path) -> list[str]:
    lines = [line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines()]
    seqs: list[str] = []
    fasta_chunks: list[str] = []
    seen_header = False
    for line in lines:
        if not line:
            continue
        if line.startswith(">"):
            seen_header = True
            if fasta_chunks:
                seqs.append("".join(fasta_chunks))
                fasta_chunks = []
            continue
        cleaned = "".join(ch for ch in line if ch in "ACGT")
        if not cleaned:
            continue
        if seen_header:
            fasta_chunks.append(cleaned)
        else:
            seqs.append(cleaned)
    if seen_header and fasta_chunks:
        seqs.append("".join(fasta_chunks))
    return [seq for seq in seqs if len(seq) >= 3 and len(seq) % 3 == 0]


def codon_sequence_to_aa_and_codons(cds: str) -> tuple[str, list[str]]:
    codons = [cds[i : i + 3] for i in range(0, len(cds), 3)]
    if codons and GENETIC_CODE.get(codons[-1]) == "*":
        codons = codons[:-1]
    if not codons:
        raise ValueError("Sequence has no sense codons after trimming terminal stop.")
    aa_chars: list[str] = []
    for codon in codons:
        aa = GENETIC_CODE.get(codon)
        if aa is None:
            raise ValueError(f"Unknown codon: {codon}")
        if aa == "*":
            raise ValueError("Internal stop codon encountered inside CDS.")
        aa_chars.append(aa)
    return "".join(aa_chars), codons


def build_cai_weights(host_usage: dict[str, float]) -> np.ndarray:
    weights = np.full(N_CODONS, np.nan, dtype=np.float64)
    synonymous = build_synonymous_codons()
    eps = 1e-12
    for codons in synonymous.values():
        usage_vals = np.array([max(host_usage.get(c, 0.0), 0.0) for c in codons]) + eps
        max_usage = float(usage_vals.max())
        for codon, value in zip(codons, usage_vals):
            weights[codon_to_index(codon)] = value / max_usage
    return weights


def sequence_cai(codons: list[str], cai_weights: np.ndarray) -> float:
    vals = np.array([cai_weights[codon_to_index(codon)] for codon in codons])
    ok = np.isfinite(vals) & (vals > 0.0)
    if not np.any(ok):
        return float("nan")
    return float(np.exp(np.mean(np.log(vals[ok]))))


@dataclass
class HostTensors:
    usage: torch.Tensor
    mu: torch.Tensor
    u_max: torch.Tensor
    u_min: torch.Tensor


@dataclass
class Metrics:
    loss: float
    mae: float
    mse: float
    pearson: float
    cai: float


def make_host_tensors(
    aa_sequence: str,
    synonymous_codons: dict[str, list[str]],
    codon_usage: dict[str, float],
) -> HostTensors:
    length = len(aa_sequence)
    usage = torch.zeros(length, N_CODONS, dtype=torch.float32)
    mu = torch.zeros(length, dtype=torch.float32)
    u_max = torch.zeros(length, dtype=torch.float32)
    u_min = torch.zeros(length, dtype=torch.float32)
    for t, aa in enumerate(aa_sequence):
        codons = synonymous_codons[aa]
        vals = torch.tensor([codon_usage[c] for c in codons], dtype=torch.float32)
        for codon, value in zip(codons, vals):
            usage[t, codon_to_index(codon)] = value
        mu[t] = vals.mean()
        u_max[t] = vals.max()
        u_min[t] = vals.min()
    return HostTensors(usage=usage, mu=mu, u_max=u_max, u_min=u_min)


def host_to_device(host: HostTensors, device: torch.device) -> HostTensors:
    return HostTensors(
        usage=host.usage.to(device),
        mu=host.mu.to(device),
        u_max=host.u_max.to(device),
        u_min=host.u_min.to(device),
    )


def hard_minmax_profile(
    chosen_codons: list[str],
    aa_sequence: str,
    synonymous_codons: dict[str, list[str]],
    codon_usage: dict[str, float],
    window_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    usage_per_pos = torch.tensor([codon_usage[codon] for codon in chosen_codons], dtype=torch.float32)
    stats = make_host_tensors(aa_sequence, synonymous_codons, codon_usage)
    delta = usage_per_pos - stats.mu
    delta_w = delta.unfold(0, window_size, 1).sum(dim=1)
    d_plus_w = (stats.u_max - stats.mu).unfold(0, window_size, 1).sum(dim=1)
    d_minus_w = (stats.mu - stats.u_min).unfold(0, window_size, 1).sum(dim=1)
    denom = torch.where(delta_w >= 0.0, d_plus_w, d_minus_w) + eps
    return 100.0 * delta_w / denom


def target_profile_tensor(
    aa_sequence: str,
    target_codons: list[str],
    synonymous_codons: dict[str, list[str]],
    target_usage: dict[str, float],
    window_size: int,
) -> torch.Tensor:
    return hard_minmax_profile(
        target_codons,
        aa_sequence,
        synonymous_codons,
        target_usage,
        window_size,
    )


def profile_window_mask(lengths: torch.Tensor, window_size: int, n_windows: int) -> torch.Tensor:
    starts = torch.arange(n_windows, device=lengths.device).unsqueeze(0)
    return (starts + window_size) <= lengths.unsqueeze(1)


def align_target_profile_to_positions(
    target_profiles: torch.Tensor,
    lengths: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    aligned = F.interpolate(
        target_profiles.unsqueeze(1),
        size=max_len,
        mode="linear",
        align_corners=False,
    ).squeeze(1)
    pos_idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    valid = pos_idx < lengths.unsqueeze(1)
    return aligned * valid.to(dtype=aligned.dtype)


def gather_target_profile_neighbors(
    target_profiles: torch.Tensor,
    lengths: torch.Tensor,
    max_len: int,
    window_size: int,
    mm_radius: int,
) -> torch.Tensor:
    batch_size, max_windows = target_profiles.shape
    device = target_profiles.device
    context_len = 2 * mm_radius + 1
    offsets = torch.arange(-mm_radius, mm_radius + 1, device=device)
    pos_t = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
    n_windows = (lengths - window_size + 1).clamp(min=0)
    lo_w = (pos_t - window_size + 1).clamp(min=0)
    hi_w = torch.minimum(pos_t, (n_windows - 1).view(-1, 1))
    center_w = (lo_w + hi_w) // 2
    gather_w = center_w.unsqueeze(-1) + offsets.view(1, 1, -1)
    pos_valid = pos_t.unsqueeze(-1) < lengths.view(-1, 1, 1)
    window_valid = (gather_w >= 0) & (gather_w < n_windows.view(-1, 1, 1))
    valid = pos_valid & window_valid
    gather_w_clamped = gather_w.clamp(0, max(max_windows - 1, 0))
    batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(
        -1, max_len, context_len
    )
    gathered = target_profiles[batch_idx, gather_w_clamped]
    return gathered * valid.to(dtype=gathered.dtype)


def target_profile_position_features(
    target_profiles: torch.Tensor,
    lengths: torch.Tensor,
    max_len: int,
    *,
    window_size: int,
    mm_radius: int,
) -> torch.Tensor:
    if mm_radius <= 0:
        return align_target_profile_to_positions(target_profiles, lengths, max_len).unsqueeze(-1)
    return gather_target_profile_neighbors(
        target_profiles,
        lengths,
        max_len,
        window_size,
        mm_radius,
    )


def smooth_minmax_profile_batched(
    probs: torch.Tensor,
    host: HostTensors,
    window_size: int,
    beta: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    expected_usage = (probs * host.usage).sum(dim=-1)
    delta = expected_usage - host.mu
    delta_w = delta.unfold(1, window_size, 1).sum(dim=-1)
    d_plus_w = (host.u_max - host.mu).unfold(1, window_size, 1).sum(dim=-1)
    d_minus_w = (host.mu - host.u_min).unfold(1, window_size, 1).sum(dim=-1)
    gate = torch.sigmoid(beta * delta_w)
    denom = gate * d_plus_w + (1.0 - gate) * d_minus_w + eps
    return 100.0 * delta_w / denom


def masked_profile_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask
    return diff.sum() / mask.sum().clamp(min=1)


def masked_profile_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff_sq = (pred - target).pow(2) * mask
    return diff_sq.sum() / mask.sum().clamp(min=1)


def masked_profile_pearson(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    valid = mask.to(dtype=pred.dtype)
    count = valid.sum().clamp(min=1.0)
    pred_mean = (pred * valid).sum() / count
    target_mean = (target * valid).sum() / count
    pred_centered = (pred - pred_mean) * valid
    target_centered = (target - target_mean) * valid
    numerator = (pred_centered * target_centered).sum()
    denom = torch.sqrt(pred_centered.pow(2).sum() * target_centered.pow(2).sum()).clamp(min=eps)
    return numerator / denom


def profile_sufficient_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float, float, float, float, float]:
    valid = mask.to(dtype=pred.dtype)
    pred_valid = pred * valid
    target_valid = target * valid
    return (
        float(valid.sum().item()),
        float(pred_valid.sum().item()),
        float(target_valid.sum().item()),
        float((pred.pow(2) * valid).sum().item()),
        float((target.pow(2) * valid).sum().item()),
        float((pred * target * valid).sum().item()),
    )


def pearson_from_sufficient_stats(
    count: float,
    pred_sum: float,
    target_sum: float,
    pred_sq_sum: float,
    target_sq_sum: float,
    cross_sum: float,
    eps: float = 1e-12,
) -> float:
    if count <= 1:
        return 0.0
    pred_var = pred_sq_sum - (pred_sum * pred_sum) / count
    target_var = target_sq_sum - (target_sum * target_sum) / count
    denom = float(np.sqrt(max(pred_var, 0.0) * max(target_var, 0.0)))
    if denom <= eps:
        return 0.0
    cov = cross_sum - (pred_sum * target_sum) / count
    return cov / denom


def decode_argmax_codons(
    probs: torch.Tensor,
    aa_sequence: str,
    synonymous_codons: dict[str, list[str]],
) -> list[str]:
    decoded: list[str] = []
    for t, aa in enumerate(aa_sequence):
        codons = synonymous_codons[aa]
        indices = [codon_to_index(codon) for codon in codons]
        idx = int(torch.argmax(probs[t, indices]).item())
        decoded.append(codons[idx])
    return decoded


@torch.no_grad()
def evaluate(
    model,
    loader,
    *,
    syn_mask: torch.Tensor,
    synonymous_codons: dict[str, list[str]],
    cai_weights: np.ndarray,
    window_size: int,
    beta: float,
    device: torch.device,
    desc: str,
) -> Metrics:
    model.eval()
    syn_mask = syn_mask.to(device)
    total_mae = 0.0
    total_mse = 0.0
    total_windows = 0.0
    pred_sum = 0.0
    target_sum = 0.0
    pred_sq_sum = 0.0
    target_sq_sum = 0.0
    cross_sum = 0.0
    total_cai = 0.0
    cai_count = 0

    for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
        aa = batch["aa"].to(device)
        lengths = batch["lengths"].to(device)
        target_profiles = batch["target_profiles"].to(device)
        mask = batch["mask"].to(device)
        host = host_to_device(batch["host"], device)

        probs = model.masked_probs(aa, lengths, target_profiles, syn_mask)
        pred_profiles = smooth_minmax_profile_batched(probs, host, window_size, beta)
        mae = masked_profile_mae(pred_profiles, target_profiles, mask)
        mse = masked_profile_mse(pred_profiles, target_profiles, mask)
        stats = profile_sufficient_stats(pred_profiles, target_profiles, mask)
        n_windows = stats[0]

        total_mae += mae.item() * n_windows
        total_mse += mse.item() * n_windows
        total_windows += n_windows
        pred_sum += stats[1]
        target_sum += stats[2]
        pred_sq_sum += stats[3]
        target_sq_sum += stats[4]
        cross_sum += stats[5]

        for b, aa_sequence in enumerate(batch["aa_strings"]):
            seq_len = int(lengths[b].item())
            pred_codons = decode_argmax_codons(
                probs[b, :seq_len].cpu(),
                aa_sequence,
                synonymous_codons,
            )
            cai = sequence_cai(pred_codons, cai_weights)
            if np.isfinite(cai):
                total_cai += cai
                cai_count += 1

    denom = max(total_windows, 1.0)
    mae = total_mae / denom
    return Metrics(
        loss=mae,
        mae=mae,
        mse=total_mse / denom,
        pearson=pearson_from_sufficient_stats(
            total_windows,
            pred_sum,
            target_sum,
            pred_sq_sum,
            target_sq_sum,
            cross_sum,
        ),
        cai=total_cai / max(cai_count, 1),
    )
