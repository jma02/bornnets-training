"""Build the fixed operators for the far-field and near-field setups in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import hankel1

WAVENUMBER = 16.0


def image_points(axis: np.ndarray) -> np.ndarray:
    x, y = np.meshgrid(axis, axis)
    return np.column_stack((x.ravel(), y.ravel()))


def farfield() -> tuple[np.ndarray, np.ndarray, float]:
    angles = np.linspace(-np.pi, np.pi, 100)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    # Row 100 * i + j has observation direction i and incident direction j.
    differences = (directions[None] - directions[:, None]).reshape(-1, 2)
    phases = differences @ image_points(np.linspace(-1, 1, 100)).T
    born = (2 / 100) ** 2 * np.exp(1j * np.pi / 4) * np.sqrt(WAVENUMBER**3 / (8 * np.pi)) * np.exp(1j * WAVENUMBER * phases)
    # Use the default cutoff of torch.linalg.pinv in single precision, which keeps 559 singular values.
    inverse = np.linalg.pinv(born, rcond=max(born.shape) * np.finfo(np.float32).eps)
    return np.vstack((born.real, born.imag)), np.hstack((inverse.real, -inverse.imag)), 1.0


def nearfield(data: Path) -> tuple[np.ndarray, np.ndarray, float]:
    step = 2 / 100
    angles = 2 * np.pi * np.arange(100) / 100
    antennas = 1.8 * np.column_stack((np.cos(angles), np.sin(angles)))
    points = image_points(np.linspace(-1 + step / 2, 1 - step / 2, 100))
    green = 0.25j * hankel1(0, WAVENUMBER * np.linalg.norm(antennas[:, None] - points, axis=2))
    # Row 100 * i + j has source i and receiver j. Each antenna is both a source and a receiver.
    born = (WAVENUMBER**2 * step**2 * green[:, None] * green[None]).reshape(10000, 10000)
    forward = np.vstack((born.real, born.imag))
    eigenvalues, vectors = np.linalg.eigh(forward.T @ forward)
    with np.load(data / "val.npz") as saved:
        fields, truth = saved["fields"], saved["truth"]
    # Compute the errors in the eigenbasis of the Gram matrix. The change of basis preserves norms.
    projected = np.hstack((fields.real, fields.imag)).astype(np.float64) @ forward @ vectors
    target = truth.reshape(len(truth), -1).astype(np.float64) @ vectors
    norms = np.linalg.norm(target, axis=1)
    gamma = min(
        (np.logspace(-5, -0.5, 10) * np.sqrt(eigenvalues[-1])) ** 2,
        key=lambda value: np.mean(np.linalg.norm(projected / (eigenvalues + value) - target, axis=1) / norms),
    )
    inverse = (vectors / (eigenvalues + gamma)) @ (forward @ vectors).T
    with np.load(data / "train.npz") as saved:
        fields = saved["fields"]
    scale = np.sqrt(np.mean(np.square(np.stack((fields.real, fields.imag))), dtype=np.float64))
    return forward, inverse, float(scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setup", choices=("farfield", "nearfield"))
    parser.add_argument("--data", type=Path, help="Near-field directory containing train.npz and val.npz")
    parser.add_argument("--output", type=Path, required=True, help="Operator file to write")
    args = parser.parse_args()
    if args.setup == "nearfield" and args.data is None:
        parser.error("nearfield requires --data")
    forward, inverse, scale = farfield() if args.setup == "farfield" else nearfield(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, forward=forward.astype(np.float32), inverse=inverse.astype(np.float32), data_scale=scale)


if __name__ == "__main__":
    main()
