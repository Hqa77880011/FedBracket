import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .vision import classification_metrics


METHODS = ["fedbracket", "fedavg", "variance", "random", "fedprox", "fedadam", "exact_bracket"]


class Model(nn.Module):
    def __init__(self, cfg, method):
        super().__init__()
        self.method = method
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg["input_dim"], cfg["hidden"]),
            nn.GELU(),
            nn.Linear(cfg["hidden"], cfg["classes"]),
        )

    def forward(self, x):
        return self.network(x.float())


def flatten(parameters):
    return torch.cat([p.reshape(-1) for p in parameters])


def assign(model, vector):
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(vector[offset : offset + parameter.numel()].view_as(parameter))
            offset += parameter.numel()


def gradient(model, x, y, graph=False):
    value = F.cross_entropy(model(x), y.long())
    return torch.autograd.grad(value, tuple(model.parameters()), create_graph=graph)


def hvp(model, x, y, vector):
    gradients = gradient(model, x, y, True)
    product = (flatten(gradients) * vector).sum()
    return flatten(torch.autograd.grad(product, tuple(model.parameters())))


def quantize(vector):
    scale = vector.abs().max().clamp_min(1e-12) / 127
    return torch.round(vector / scale).clamp(-127, 127) * scale


def fit(model, train, validation, cfg, output):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = next(model.parameters()).device
    x = torch.as_tensor(train.data["x"], device=device)
    y = torch.as_tensor(train.data["y"], device=device)
    clients = np.asarray(train.data["client"], dtype=int)
    client_ids = np.unique(clients)
    if len(client_ids) < 2:
        raise ValueError("Federated training requires at least two nonempty clients")
    pools = {int(i): np.flatnonzero(clients == i) for i in client_ids}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    ema = 0.0
    total_audit = 0.0
    moments = torch.zeros_like(flatten(tuple(model.parameters())))
    variances = torch.zeros_like(moments)
    random_schedule = None
    if model.method == "random":
        reference = Path(
            cfg.get("reference_history", output.parent / "fedbracket" / "history.jsonl")
        )
        reference_rows = [
            json.loads(line) for line in reference.read_text(encoding="utf-8").splitlines()
        ]
        random_schedule = [row["local_steps"] for row in reference_rows]
        if len(random_schedule) < cfg["rounds"]:
            raise ValueError("Reference schedule is shorter than the requested round budget")
        rng.shuffle(random_schedule)
    start_total = time.perf_counter()
    for round_index in range(cfg["rounds"]):
        start = time.perf_counter()
        selected = rng.choice(
            client_ids,
            max(2, min(len(client_ids), int(np.ceil(cfg["participation"] * len(client_ids))))),
            replace=False,
        )
        global_vector = flatten(tuple(model.parameters())).detach().clone()
        if round_index % cfg["audit_interval"] == 0 and model.method in (
            "fedbracket",
            "variance",
            "exact_bracket",
        ):
            audit_start = time.perf_counter()
            energies = []
            for _ in range(cfg["audit_pairs"]):
                i, j = rng.choice(selected, 2, replace=False)
                ix = rng.choice(
                    pools[int(i)], min(len(pools[int(i)]), cfg["audit_batch_size"]), replace=False
                )
                jx = rng.choice(
                    pools[int(j)], min(len(pools[int(j)]), cfg["audit_batch_size"]), replace=False
                )
                gi, gj = (
                    flatten(gradient(model, x[ix], y[ix])).detach(),
                    flatten(gradient(model, x[jx], y[jx])).detach(),
                )
                if model.method == "variance":
                    energy = (gi - gj).square().sum()
                else:
                    mask = torch.ones_like(gi)
                    if model.method != "exact_bracket":
                        mask.zero_()
                        chosen = rng.choice(
                            len(gi), max(1, int(len(gi) * cfg["audit_fraction"])), replace=False
                        )
                        mask[torch.as_tensor(chosen, device=device)] = 1
                    di, dj = gi * mask, gj * mask
                    if model.method != "exact_bracket":
                        di, dj = quantize(di), quantize(dj)
                    bracket = hvp(model, x[jx], y[jx], di) - hvp(model, x[ix], y[ix], dj)
                    energy = (bracket * mask).square().sum() / (
                        gi.square().sum() + gj.square().sum()
                    ).clamp_min(1e-12)
                energies.append(float(energy.detach()))
            ema = cfg["ema"] * ema + (1 - cfg["ema"]) * np.mean(energies)
            total_audit += time.perf_counter() - audit_start
        steps = cfg["minimum_steps"] if ema > cfg["audit_threshold"] else cfg["local_steps"]
        if model.method in ("fedavg", "fedprox", "fedadam"):
            steps = cfg["local_steps"]
        if random_schedule is not None:
            steps = random_schedule[round_index]
        proposals, weights = [], []
        for client in selected:
            local = copy.deepcopy(model)
            optimizer = torch.optim.SGD(local.parameters(), lr=cfg["learning_rate"])
            for _ in range(steps):
                indices = rng.choice(
                    pools[int(client)],
                    min(cfg["batch_size"], len(pools[int(client)])),
                    replace=False,
                )
                optimizer.zero_grad()
                objective = F.cross_entropy(local(x[indices]), y[indices].long())
                if model.method == "fedprox":
                    objective = (
                        objective
                        + 0.5
                        * cfg["proximal_weight"]
                        * (flatten(tuple(local.parameters())) - global_vector).square().sum()
                    )
                objective.backward()
                optimizer.step()
            proposals.append(flatten(tuple(local.parameters())).detach())
            weights.append(len(pools[int(client)]))
        weight = torch.as_tensor(
            np.asarray(weights) / sum(weights), device=device, dtype=global_vector.dtype
        )
        stacked = torch.stack(proposals)
        mean = (stacked * weight[:, None]).sum(0)
        drift = float(((stacked - mean).square().sum(1) * weight).sum())
        if model.method == "fedadam":
            delta = mean - global_vector
            moments = 0.9 * moments + 0.1 * delta
            variances = 0.99 * variances + 0.01 * delta.square()
            mean = global_vector + cfg["server_learning_rate"] * (
                moments / (1 - 0.9 ** (round_index + 1))
            ) / ((variances / (1 - 0.99 ** (round_index + 1))).sqrt() + 1e-3)
        assign(model, mean)
        model.eval()
        with torch.no_grad():
            vx = torch.as_tensor(validation.data["x"], device=device)
            vy = torch.as_tensor(validation.data["y"], device=device)
            accuracy = float((model(vx).argmax(-1) == vy).float().mean())
        row = {
            "round": round_index + 1,
            "validation_accuracy": accuracy,
            "local_steps": steps,
            "bracket_ema": float(ema),
            "drift": drift,
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        torch.save({"state": model.state_dict()}, output / f"round_{round_index + 1:04d}.pt")
        if accuracy > best:
            best = accuracy
            cfg["training_method"] = model.method
            torch.save(
                {"state": model.state_dict(), "config": cfg, "method": model.method},
                output / "model.pt",
            )
        model.train()
    cfg["training_metrics"] = {
        "mean_local_steps": float(np.mean([r["local_steps"] for r in history])),
        "mean_client_drift": float(np.mean([r["drift"] for r in history])),
        "audit_seconds": total_audit,
        "training_seconds": time.perf_counter() - start_total,
    }
    payload = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
    payload["config"] = cfg
    torch.save(payload, output / "model.pt")
    (output / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
    )


def loss(model, batch, cfg):
    return F.cross_entropy(model(batch["x"]), batch["y"].long())


def predict(model, batch, cfg, calibration=None):
    return {"logits": model(batch["x"])}


def metrics(prediction, data, cfg):
    result = classification_metrics(prediction["logits"], data["y"])
    result.update(cfg.get("training_metrics", {}))
    root = Path(cfg["_checkpoint_directory"])
    curve = []
    evaluator = Model(cfg, cfg["training_method"])
    for path in sorted(root.glob("round_*.pt")):
        evaluator.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state"])
        with torch.no_grad():
            logits = evaluator(torch.tensor(data["x"]))
        curve.append(float((logits.argmax(-1).numpy() == data["y"]).mean()))
    reached = [i + 1 for i, accuracy in enumerate(curve) if accuracy >= cfg["target_accuracy"]]
    result["rounds_to_target"] = reached[0] if reached else None
    result["test_accuracy_curve"] = curve
    return result


def inference(model, batch, cfg):
    return model(batch["x"])
