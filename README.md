# FedBracket

Implementation of **Controlling federated drift through client-update noncommutativity**.

Runs paired HVP audits, quantizes sampled direction coordinates, smooths bracket energy and shortens high-bracket local trajectories before aggregation.

## Install

Use Python 3.11 or newer. Install a matching PyTorch/TorchVision build for the target CPU or CUDA device, then install the package.

```bash
git clone https://github.com/Hqa77880011/FedBracket.git
cd FedBracket
python -m venv .venv
```

Activate with `source .venv/bin/activate` or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -e .
```

## Prepare data

Use registered non-IID CIFAR-100 client partitions or a controlled geometry dataset. Client IDs must be assigned before train/validation/test packing.

x is image/features, y is class and client the training client ID. The model is a matched MLP over flattened inputs. Validation/test examples use the common held-out classification protocol.

```bash
python -m fedbracket prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Train

Edit `configs/default.json` for the input dimensions, device and training budget.

```bash
python -m fedbracket train --config configs/default.json --data data/prepared --output runs/main
python -m fedbracket suite --config configs/default.json --data data/prepared --output runs/comparison
```

`--methods` selects a subset. fedbracket, fedavg, variance, random, fedprox, fedadam and exact_bracket. random permutes the observed FedBracket step multiset from the same suite root; standalone random requires reference_history.

The common driver uses AdamW with cosine learning-rate decay, validation-objective early stopping and a fixed seed. Custom federated/continual drivers use their registered round/task budgets. It saves only a validation-selected model and measured epoch history. `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `patience` and `device` control optimization. Use separate output roots for independent runs.

## Evaluate and visualize

```bash
python -m fedbracket evaluate --checkpoint runs/main/fedbracket/model.pt --data data/prepared/test.npz --output results/fedbracket
python -m fedbracket plot --inputs runs/main/fedbracket results/fedbracket --output figures
```

`model.pt` contains model parameters, configuration and method identity. `history.jsonl` contains measured training/validation loss and epoch time. Evaluation writes `predictions.npz` and `metrics.json`. Plotting creates loss curves and metric comparisons from supplied run directories.

Final accuracy/F1/ECE, rounds to a registered test accuracy, mean local steps, client-update drift, audit/training cost and full test accuracy curves.

## Implementation conventions

The custom driver uses a fixed communication-round budget and records each round checkpoint. Test curves are evaluated afterward and never steer training. Audits use non-secure client gradient visibility. The random schedule's prior control run is needed to match total local computation exactly.


## Pack image records

A JSONL manifest can point to images, masks and NPY feature arrays. Paths are relative to the manifest. All rows use identical keys; group stores the source object ID. For example, an image-classification row is `{"x":"images/1.png","y":0,"group":"object-1"}`.

```bash
python -m fedbracket.pack --manifest data/manifest.jsonl --output data/raw.npz --channels 3 --target class
```

## CIFAR-100 preparation

```bash
python -m fedbracket.cifar --output data/raw.npz --clients 100 --alpha 0.1
```

This downloads the author dataset through TorchVision, reserves 10% of its training images for validation, and preserves the original test set. It writes fine/coarse labels, source IDs and task metadata. Paired/audited transformation fields and ontology constraints must be added from the registered pair/ontology protocol before using those objectives.

Inference timing runs the deployment forward pass separately from calibration, commutation and other evaluator diagnostics, after one warmup. Batch P50/P95 and examples/s retain the configured batch size. MAC/FLOP counts cover Conv/Linear/GRU multiply-adds; process RSS includes the dataset and Python runtime.
