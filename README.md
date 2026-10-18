# FedBracket

Implementation of **Controlling federated drift through client-update noncommutativity**.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

The adapter downloads [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), preserves its test set, reserves 10% of training examples for validation and creates 100 Dirichlet clients. Client assignments and original sample IDs are stored in the NPZ.

```bash
python -m fedbracket.cifar --output data/raw.npz --clients 100 --alpha 0.1 --seed 17
python -m fedbracket prepare --source data/raw.npz --output data/prepared --seed 17
```

## Training

Use the same `configs/default.json` throughout. Its seed is `17`; preserve the prepared splits and use separate output directories for new seeds.

```bash
python -m fedbracket train --config configs/default.json --data data/prepared --output runs/main
```

## Evaluation

```bash
python -m fedbracket evaluate --checkpoint runs/main/fedbracket/model.pt --data data/prepared/test.npz --output results/fedbracket
```

## Comparisons and ablations

`suite` trains the implemented methods on the same prepared split. Use `--methods` to select a subset, then evaluate each method checkpoint with the evaluation command above.

```bash
python -m fedbracket suite --config configs/default.json --data data/prepared --output runs/comparison
```

## Outputs

Each training method directory contains `model.pt` and `history.jsonl`. Evaluation writes `predictions.npz` and `metrics.json`; plots read the supplied run directories.

```bash
python -m fedbracket plot --inputs runs/main/fedbracket results/fedbracket --output figures
```

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
