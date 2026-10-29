"""Train BornNet, LinearBornNet, or BornAE on paired scattering data."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models import BornNet


def load_data(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as saved:
        fields = saved["fields"]
        truth = saved["truth"]
    if fields.ndim != 2 or fields.shape[1] != 10000 or not np.iscomplexobj(fields):
        raise ValueError(f"{path}: fields must be a complex array of shape (N, 10000)")
    if truth.shape != (len(fields), 1, 100, 100) or np.iscomplexobj(truth):
        raise ValueError(f"{path}: truth must be real contrast with shape (N, 1, 100, 100)")
    if not len(fields) or not np.isfinite(fields).all() or not np.isfinite(truth).all():
        raise ValueError(f"{path}: data must be nonempty and finite")
    channels = np.stack((fields.real, fields.imag), axis=1).reshape(-1, 2, 100, 100)
    return torch.from_numpy(channels.astype(np.float32)), torch.from_numpy(truth.astype(np.float32))


@torch.inference_mode()
def evaluate(
    model: BornNet,
    fields: torch.Tensor,
    truth: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> float:
    model.eval()
    total = 0.0
    for inputs, targets in zip(fields.split(batch_size), truth.split(batch_size)):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            image, _ = model.encode(inputs)
        errors = torch.linalg.vector_norm((image.float() - targets).flatten(1), dim=1)
        norms = torch.linalg.vector_norm(targets.flatten(1), dim=1)
        total += (errors / norms).sum().item()
    error = total / len(fields)
    if not math.isfinite(error):
        raise FloatingPointError("Nonfinite validation error; targets must have nonzero contrast norm")
    return error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("BornNet", "LinearBornNet", "BornAE"), required=True)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing train.npz and val.npz")
    parser.add_argument("--operators", type=Path, help="NPZ with inverse, forward, and optional data_scale")
    parser.add_argument("--data-scale", type=float, help="Input scale; overrides data_scale from --operators (default 1)")
    parser.add_argument("--output", type=Path, required=True, help="New directory for best.pt")
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Run learned layers under bfloat16 autocast (CUDA only)")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.threads) < 1:
        parser.error("steps, batch-size, eval-every, and threads must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if args.model != "LinearBornNet" and args.operators is None:
        parser.error("BornNet and BornAE require --operators")
    device = torch.device(args.device)
    if args.amp and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("--amp requires a CUDA device with bfloat16 support")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    inverse, forward, scale = None, None, 1.0
    if args.operators is not None:
        with np.load(args.operators, allow_pickle=False) as saved:
            scale = float(saved.get("data_scale", 1.0))
            key = {"BornNet": "inverse", "BornAE": "forward"}.get(args.model)
            if key is not None:
                matrix = saved[key]
                shape = (10000, 20000) if key == "inverse" else (20000, 10000)
                if matrix.shape != shape or np.iscomplexobj(matrix) or not np.isfinite(matrix).all():
                    raise ValueError(f"{key} must be a finite real matrix with shape {shape}")
                operator = torch.from_numpy(matrix.astype(np.float32))
                if key == "inverse":
                    inverse = operator
                else:
                    forward = operator
    if args.data_scale is not None:
        scale = args.data_scale
    if not math.isfinite(scale) or scale <= 0:
        parser.error("data-scale must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=False)
    model = BornNet(args.model, inverse=inverse, forward=forward, data_scale=scale).to(device)
    train_fields, train_truth = load_data(args.data / "train.npz")
    val_fields, val_truth = load_data(args.data / "val.npz")
    if args.batch_size > len(train_fields):
        parser.error("batch-size exceeds the number of training objects")
    train_fields.div_(scale)
    val_fields.div_(scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    batches = torch.Generator().manual_seed(100000 + args.seed)
    best = float("inf")
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.randperm(len(train_fields), generator=batches)[:args.batch_size]
        fields, truth = train_fields[indices].to(device), train_truth[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            image, auxiliary = model(fields)
        loss = F.mse_loss(image.float(), truth)
        if args.model == "BornAE":
            loss = loss + scale**2 * F.mse_loss(auxiliary.float(), fields)
            loss = loss + 1e-4 * image.float().abs().mean()
        else:
            loss = loss + F.mse_loss(auxiliary.float(), truth)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            error = evaluate(model, val_fields, val_truth, args.batch_size, device, args.amp)
            if error < best:
                best = error
                torch.save({
                    "model": model.state_dict(), "kind": args.model,
                    "data_scale": scale, "step": step, "validation_error": error,
                }, args.output / "best.pt")
            print(f"step={step} loss={loss.item():.6g} validation={error:.6f} best={best:.6f}", flush=True)


if __name__ == "__main__":
    main()
