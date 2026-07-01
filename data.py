from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import (
    N_CODONS,
    HostTensors,
    codon_sequence_to_aa_and_codons,
    encode_aa_sequence,
    make_host_tensors,
    parse_sequences,
    profile_window_mask,
    target_profile_tensor,
)


@dataclass
class ProfileSample:
    aa_sequence: str
    target_codons: list[str]


class ProfileMatchDataset(Dataset):
    def __init__(self, samples: list[ProfileSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ProfileSample:
        return self.samples[idx]


def collate_profile_batch(
    batch: list[ProfileSample],
    synonymous_codons: dict[str, list[str]],
    target_usage: dict[str, float],
    host_usage: dict[str, float],
    window_size: int,
) -> dict:
    aa_strings = [sample.aa_sequence for sample in batch]
    lengths = torch.tensor([len(s) for s in aa_strings], dtype=torch.long)
    max_len = int(lengths.max().item())
    max_windows = max(max_len - window_size + 1, 0)

    aa_batch = torch.zeros(len(batch), max_len, dtype=torch.long)
    host_usage_batch = torch.zeros(len(batch), max_len, N_CODONS, dtype=torch.float32)
    host_mu = torch.zeros(len(batch), max_len, dtype=torch.float32)
    host_u_max = torch.zeros(len(batch), max_len, dtype=torch.float32)
    host_u_min = torch.zeros(len(batch), max_len, dtype=torch.float32)
    target_profiles = torch.zeros(len(batch), max_windows, dtype=torch.float32)

    for i, sample in enumerate(batch):
        aa_sequence = sample.aa_sequence
        seq_len = len(aa_sequence)
        aa_batch[i, :seq_len] = encode_aa_sequence(aa_sequence)

        host = make_host_tensors(aa_sequence, synonymous_codons, host_usage)
        host_usage_batch[i, :seq_len] = host.usage
        host_mu[i, :seq_len] = host.mu
        host_u_max[i, :seq_len] = host.u_max
        host_u_min[i, :seq_len] = host.u_min

        profile = target_profile_tensor(
            aa_sequence,
            sample.target_codons,
            synonymous_codons,
            target_usage,
            window_size,
        )
        n_windows = seq_len - window_size + 1
        target_profiles[i, :n_windows] = profile

    return {
        "aa": aa_batch,
        "aa_strings": aa_strings,
        "lengths": lengths,
        "host": HostTensors(
            usage=host_usage_batch,
            mu=host_mu,
            u_max=host_u_max,
            u_min=host_u_min,
        ),
        "target_profiles": target_profiles,
        "target_codons": [sample.target_codons for sample in batch],
        "mask": profile_window_mask(lengths, window_size, max_windows),
    }


def make_loader(
    samples: list[ProfileSample],
    *,
    synonymous_codons: dict[str, list[str]],
    target_usage: dict[str, float],
    host_usage: dict[str, float],
    window_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    collate_fn = partial(
        collate_profile_batch,
        synonymous_codons=synonymous_codons,
        target_usage=target_usage,
        host_usage=host_usage,
        window_size=window_size,
    )
    return DataLoader(
        ProfileMatchDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def _cache_path(cds_path: Path) -> Path:
    resolved = cds_path.resolve()
    return resolved.parent / ".cache" / f"{resolved.stem}.minimal_profile_samples.pkl"


def parse_cds_profile_samples(cds_path: Path, refresh_cache: bool = False) -> list[ProfileSample]:
    cache_path = _cache_path(cds_path)
    fingerprint = (str(cds_path.resolve()), cds_path.stat().st_mtime_ns, cds_path.stat().st_size)
    if not refresh_cache and cache_path.is_file():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("fingerprint") == fingerprint:
                samples = payload.get("samples")
                if isinstance(samples, list):
                    print(f"Loaded {len(samples)} CDS records from cache: {cache_path}")
                    return samples
        except (OSError, pickle.UnpicklingError, TypeError, ValueError):
            pass

    sequences = parse_sequences(cds_path)
    samples: list[ProfileSample] = []
    for cds in tqdm(sequences, desc="Translating human CDS", unit="seq"):
        try:
            aa_sequence, target_codons = codon_sequence_to_aa_and_codons(cds)
        except ValueError:
            continue
        samples.append(ProfileSample(aa_sequence=aa_sequence, target_codons=target_codons))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump({"fingerprint": fingerprint, "samples": samples}, handle)
    print(f"Parsed {len(samples)} CDS records; cache saved to {cache_path}")
    return samples


def chunk_profile_sample(
    sample: ProfileSample,
    *,
    chunk_size: int,
    window_size: int,
    min_codons: int,
) -> list[ProfileSample]:
    n = len(sample.aa_sequence)
    if n <= chunk_size:
        return [sample]
    overlap = window_size - 1
    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError(f"chunk_size ({chunk_size}) must be > window_size - 1 ({overlap})")

    chunks: list[ProfileSample] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end - start >= max(min_codons, window_size):
            chunks.append(
                ProfileSample(
                    aa_sequence=sample.aa_sequence[start:end],
                    target_codons=sample.target_codons[start:end],
                )
            )
        if end >= n:
            break
        start += stride
    return chunks


def filter_and_chunk_samples(
    samples: list[ProfileSample],
    *,
    min_codons: int,
    max_codons: int,
    chunk_size: int,
    window_size: int,
) -> list[ProfileSample]:
    filtered: list[ProfileSample] = []
    n_short = 0
    n_long = 0
    for sample in samples:
        length = len(sample.aa_sequence)
        if length < min_codons:
            n_short += 1
            continue
        if length > max_codons:
            n_long += 1
            continue
        filtered.append(sample)

    chunked: list[ProfileSample] = []
    n_split = 0
    for sample in filtered:
        pieces = chunk_profile_sample(
            sample,
            chunk_size=chunk_size,
            window_size=window_size,
            min_codons=min_codons,
        )
        n_split += int(len(pieces) > 1)
        chunked.extend(pieces)

    print(
        f"Length filter kept {len(filtered)} / {len(samples)} CDS "
        f"(dropped {n_short} short, {n_long} long)."
    )
    print(
        f"Chunking produced {len(chunked)} samples "
        f"(chunk_size={chunk_size}, overlap={window_size - 1}, split_sources={n_split})."
    )
    return chunked


def train_test_split(
    samples: list[ProfileSample],
    *,
    rng: random.Random,
    data_fraction: float,
    test_fraction: float,
) -> tuple[list[ProfileSample], list[ProfileSample]]:
    if not 0.0 < data_fraction <= 1.0:
        raise ValueError(f"data_fraction must be in (0, 1], got {data_fraction}")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    samples = list(samples)
    rng.shuffle(samples)
    subset_size = max(2, int(len(samples) * data_fraction))
    subset = samples[:subset_size]
    n_test = max(1, int(round(subset_size * test_fraction)))
    n_train = subset_size - n_test
    if n_train < 1:
        raise ValueError("Not enough samples after splitting.")
    return subset[:n_train], subset[n_train:]
