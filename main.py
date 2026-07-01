from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from data import filter_and_chunk_samples, make_loader, parse_cds_profile_samples, train_test_split
from model import CodonProfileNet
from utils import (
    Metrics,
    build_cai_weights,
    build_synonymous_codons,
    build_synonymous_mask,
    decode_argmax_codons,
    evaluate,
    host_to_device,
    masked_profile_mae,
    masked_profile_mse,
    pearson_from_sufficient_stats,
    parse_codon_usage_csv,
    profile_sufficient_stats,
    sequence_cai,
    smooth_minmax_profile_batched,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    parser = argparse.ArgumentParser(
        description="Minimal human-to-E. coli Smooth %MinMax trainer"
    )
    parser.add_argument("--cds-path", type=Path, default=data_dir / "cds_human.txt")
    parser.add_argument(
        "--target-usage",
        type=Path,
        default=data_dir / "codon_usage_9606_human.csv",
    )
    parser.add_argument(
        "--host-usage",
        type=Path,
        default=data_dir / "codon_usage_511145_ecoli.csv",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--mm-radius", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--min-codons", type=int, default=30)
    parser.add_argument("--max-codons", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=root / "results")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.window_size < 1:
        raise ValueError("--window-size must be >= 1")
    if args.mm_radius < 0:
        raise ValueError("--mm-radius must be >= 0")
    if args.min_codons < args.window_size:
        raise ValueError("--min-codons must be >= --window-size")
    if args.max_codons < args.min_codons:
        raise ValueError("--max-codons must be >= --min-codons")
    if args.chunk_size < args.window_size:
        raise ValueError("--chunk-size must be >= --window-size")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.lower() == "cpu":
        return torch.device("cpu")
    return torch.device(device_arg)


def train_one_epoch(
    model: CodonProfileNet,
    loader,
    optimizer: torch.optim.Optimizer,
    *,
    syn_mask: torch.Tensor,
    window_size: int,
    beta: float,
    device: torch.device,
    desc: str,
    synonymous_codons: dict[str, list[str]],
    cai_weights,
) -> Metrics:
    model.train()
    syn_mask = syn_mask.to(device)
    total_loss = 0.0
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

        optimizer.zero_grad()
        probs = model.masked_probs(aa, lengths, target_profiles, syn_mask)
        pred_profiles = smooth_minmax_profile_batched(probs, host, window_size, beta)
        loss = masked_profile_mae(pred_profiles, target_profiles, mask)
        mse = masked_profile_mse(pred_profiles, target_profiles, mask)
        stats = profile_sufficient_stats(pred_profiles.detach(), target_profiles, mask)
        n_windows = stats[0]
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * n_windows
        total_mae += loss.item() * n_windows
        total_mse += mse.item() * n_windows
        total_windows += n_windows
        pred_sum += stats[1]
        target_sum += stats[2]
        pred_sq_sum += stats[3]
        target_sq_sum += stats[4]
        cross_sum += stats[5]

        with torch.no_grad():
            for b, aa_sequence in enumerate(batch["aa_strings"]):
                seq_len = int(lengths[b].item())
                pred_codons = decode_argmax_codons(
                    probs[b, :seq_len].detach().cpu(),
                    aa_sequence,
                    synonymous_codons,
                )
                cai = sequence_cai(pred_codons, cai_weights)
                if math.isfinite(cai):
                    total_cai += cai
                    cai_count += 1

    denom = max(total_windows, 1.0)
    mae = total_mae / denom
    return Metrics(
        loss=total_loss / denom,
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


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_mae",
                "train_mse",
                "train_pearson",
                "train_cai",
                "test_loss",
                "test_mae",
                "test_mse",
                "test_pearson",
                "test_cai",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_run_dir(out_dir: Path) -> Path:
    run_dir = out_dir / datetime.now().strftime("human_to_ecoli_%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    run_dir = build_run_dir(args.out_dir)
    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    print("Dataset: human CDS target profile -> E. coli host codon usage")

    synonymous_codons = build_synonymous_codons()
    syn_mask = build_synonymous_mask()
    target_usage = parse_codon_usage_csv(args.target_usage)
    host_usage = parse_codon_usage_csv(args.host_usage)
    cai_weights = build_cai_weights(host_usage)

    samples = parse_cds_profile_samples(args.cds_path, refresh_cache=args.refresh_cache)
    samples = filter_and_chunk_samples(
        samples,
        min_codons=args.min_codons,
        max_codons=args.max_codons,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
    )
    train_samples, test_samples = train_test_split(
        samples,
        rng=rng,
        data_fraction=args.data_fraction,
        test_fraction=args.test_fraction,
    )
    print(f"Train samples: {len(train_samples)}, test samples: {len(test_samples)}")

    train_loader = make_loader(
        train_samples,
        synonymous_codons=synonymous_codons,
        target_usage=target_usage,
        host_usage=host_usage,
        window_size=args.window_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_samples,
        synonymous_codons=synonymous_codons,
        target_usage=target_usage,
        host_usage=host_usage,
        window_size=args.window_size,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = CodonProfileNet(
        window_size=args.window_size,
        mm_radius=args.mm_radius,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    rows: list[dict] = []
    best_test_mae = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            syn_mask=syn_mask,
            window_size=args.window_size,
            beta=args.beta,
            device=device,
            desc=f"Epoch {epoch}/{args.epochs} train",
            synonymous_codons=synonymous_codons,
            cai_weights=cai_weights,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            syn_mask=syn_mask,
            synonymous_codons=synonymous_codons,
            cai_weights=cai_weights,
            window_size=args.window_size,
            beta=args.beta,
            device=device,
            desc=f"Epoch {epoch}/{args.epochs} test",
        )
        if test_metrics.mae < best_test_mae:
            best_test_mae = test_metrics.mae
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_mae": train_metrics.mae,
            "train_mse": train_metrics.mse,
            "train_pearson": train_metrics.pearson,
            "train_cai": train_metrics.cai,
            "test_loss": test_metrics.loss,
            "test_mae": test_metrics.mae,
            "test_mse": test_metrics.mse,
            "test_pearson": test_metrics.pearson,
            "test_cai": test_metrics.cai,
        }
        rows.append(row)
        write_metrics_csv(run_dir / "metrics.csv", rows)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_mae={train_metrics.mae:.4f} "
            f"test_mae={test_metrics.mae:.4f} "
            f"test_mse={test_metrics.mse:.4f} "
            f"test_r={test_metrics.pearson:.4f} "
            f"test_cai={test_metrics.cai:.4f}"
        )

    final = rows[-1]
    summary = {
        "dataset": "human_to_ecoli_heterologous",
        "cds_path": str(args.cds_path),
        "target_usage": str(args.target_usage),
        "host_usage": str(args.host_usage),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "window_size": args.window_size,
        "mm_radius": args.mm_radius,
        "beta": args.beta,
        "min_codons": args.min_codons,
        "max_codons": args.max_codons,
        "chunk_size": args.chunk_size,
        "data_fraction": args.data_fraction,
        "test_fraction": args.test_fraction,
        "num_train_samples": len(train_samples),
        "num_test_samples": len(test_samples),
        "best_epoch": best_epoch,
        "best_test_mae": best_test_mae,
        "final_test_mae": final["test_mae"],
        "final_test_mse": final["test_mse"],
        "final_test_pearson": final["test_pearson"],
        "final_test_cai": final["test_cai"],
        "seed": args.seed,
        "device": str(device),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
