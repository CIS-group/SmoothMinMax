# SmoothMinMax

Minimal PyTorch trainer for learning codon choices with a differentiable Smooth
%MinMax profile-matching objective.

This version is intentionally small and focused: it trains a neural codon
predictor for a fixed heterologous setting, using human CDS sequences as the
origin profile source and E. coli codon usage as the host usage table.

## What This Trains

The model receives:

- an amino-acid sequence,
- the target hard %MinMax profile computed from the native human CDS codons,
- optional neighboring %MinMax context controlled by `--mm-radius`.

It predicts a synonymous-codon probability distribution at each amino-acid
position. The predicted distribution is converted into a smooth %MinMax profile,
and training minimizes profile MAE against the target profile.

The core loss is:

```text
loss = MAE(smooth_predicted_%MinMax_profile, target_%MinMax_profile)
```

## Repository Layout

```text
SmoothMinMax/
  main.py      # training entry point
  model.py     # BiLSTM codon-profile model
  data.py      # human CDS loading, filtering, chunking, DataLoader
  utils.py     # codon utilities, %MinMax functions, metrics, evaluation
  data/
    cds_human.txt
    codon_usage_9606_human.csv
    codon_usage_511145_ecoli.csv
```

## Data

The default paths assume the required files are already present in
`SmoothMinMax/data/`:

- `cds_human.txt`: human CDS sequences, one nucleotide sequence per line
- `codon_usage_9606_human.csv`: human codon usage table
- `codon_usage_511145_ecoli.csv`: E. coli codon usage table

The trainer is heterologous-only:

```text
target/origin usage = human
host usage          = E. coli
```

There is no random dataset mode and no homologous mode in this minimal version.

## Installation

Required Python packages:

```text
torch
numpy
tqdm
```

Install them in your preferred environment, for example:

```bash
pip install torch numpy tqdm
```

## Run

From the repository root:

```bash
python SmoothMinMax/main.py
```

Example with explicit settings:

```bash
python SmoothMinMax/main.py \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --window-size 10 \
  --mm-radius 1 \
  --beta 0.1 \
  --min-codons 30 \
  --max-codons 2048 \
  --chunk-size 512 \
  --data-fraction 1.0
```

## Important Options

```text
--window-size       Sliding window size for %MinMax profiles.
--mm-radius         Number of neighboring target-profile windows exposed per codon.
                    0 uses a single aligned target-profile value.
                    r > 0 uses 2r + 1 neighboring window values.
--beta              Sigmoid sharpness for the smooth min/max denominator gate.
--min-codons        Drop CDS records shorter than this many codons.
--max-codons        Drop CDS records longer than this many codons.
--chunk-size        Split retained CDS records into overlapping chunks.
--data-fraction     Fraction of filtered/chunked samples to use.
--test-fraction     Fraction of the selected subset held out for testing.
```

Chunk overlap is `window_size - 1`, so every %MinMax window is fully contained in
at least one chunk.

## Metrics

Each epoch writes:

```text
epoch
train_loss
train_mae
train_mse
train_pearson
train_cai
test_loss
test_mae
test_mse
test_pearson
test_cai
```

Profile metrics are computed on soft predicted %MinMax profiles:

- `MAE`: mean absolute error against the target profile
- `MSE`: mean squared error against the target profile
- `Pearson`: global Pearson correlation over valid profile windows

`CAI` is computed after argmax decoding the predicted synonymous codons and
scoring the decoded sequence with the E. coli host codon usage table.

## Outputs

Runs are saved under:

```text
SmoothMinMax/results/human_to_ecoli_<timestamp>/
```

Each run contains:

```text
metrics.csv      # epoch-level metrics
summary.json     # config and final/best metrics
best_model.pt    # checkpoint with best test MAE
model.pt         # final model checkpoint
```

## Notes

This is a deliberately minimal experiment implementation. It omits the larger
experimental options from earlier scripts, including random datasets, homologous
mode, entropy regularization, learning-rate scheduling, early stopping,
gradient clipping, and plotting.

