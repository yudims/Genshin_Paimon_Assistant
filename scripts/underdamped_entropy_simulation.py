"""Simulation utilities for estimating entropy production in an underdamped Langevin system.

This module reproduces the experiment performed in the original research notebook, but in a
stand-alone Python script that can be executed directly.  The script simulates a collection of
underdamped Langevin trajectories and compares the entropy production rate estimated from the
thermodynamic uncertainty relation (TUR) against a semi-analytical reference curve.

Running this file as a script will generate a plot similar to the one shown in the notebook:

    python scripts/underdamped_entropy_simulation.py

The default configuration mirrors the notebook parameters, but the values can be changed by
modifying the constants in :func:`main` or by importing the module and calling
:func:`simulate_entropy_production` directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SimulationConfig:
    """Container describing the simulation setup."""

    dt: float = 1e-3
    total_time: float = 0.25
    n_trajectories: Tuple[int, int] = (10_000, 100_000)
    basis_size: int = 10
    regularisation_exponent: int = 2
    sampling_interval: int = 5

    @property
    def total_steps(self) -> int:
        return int(self.total_time / self.dt)


def init_state(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Draw the initial positions and momenta for ``n`` trajectories."""

    x0 = rng.normal(0.0, 0.025, size=n)
    p0 = rng.normal(0.2, 0.3, size=n)
    return x0, p0


def simulate_underdamped(
    n: int, config: SimulationConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the underdamped Langevin dynamics."""

    total_steps = config.total_steps
    dt = config.dt
    x = np.zeros((n, total_steps + 1))
    p = np.zeros_like(x)
    x[:, 0], p[:, 0] = init_state(n, rng)

    for step in range(total_steps):
        p[:, step + 1] = p[:, step] - p[:, step] * dt + math.sqrt(2 * dt) * rng.standard_normal(n)
        x[:, step + 1] = x[:, step] + p[:, step] * dt

    return x, p


def construct_basis(
    x_data: np.ndarray, p_data: np.ndarray, basis_size: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Construct Gaussian basis functions over the support of the trajectory data."""

    nx = np.linspace(x_data.min(), x_data.max(), basis_size)
    np_ = np.linspace(p_data.min(), p_data.max(), basis_size)

    # Use the spacing between neighbouring centres as the Gaussian width.  The data is spread
    # approximately evenly across the range, so this heuristic works well in practice.
    sigma_x = (nx[-1] - nx[0]) / max(len(nx) - 1, 1)
    sigma_p = (np_[-1] - np_[0]) / max(len(np_) - 1, 1)
    return nx, np_, sigma_x, sigma_p


def sigma_hat(
    x_t: np.ndarray,
    p_t: np.ndarray,
    p_prev: np.ndarray | None,
    p_next: np.ndarray | None,
    beta: float,
    nx: np.ndarray,
    np_: np.ndarray,
    sigma_x: float,
    sigma_p: float,
    dt: float,
) -> float:
    """Estimate the entropy production rate at a single time step via the TUR."""

    if p_prev is None:
        a = (x_t[:, 1] - x_t[:, 0]) + 0.5 * (p_next - p_t)
    elif p_next is None:
        a = -0.5 * (p_t - p_prev)
    else:
        a = (x_t[:, 1] - x_t[:, 0]) - 0.5 * (p_next - p_prev)

    x_kernel = np.exp(-0.5 * ((x_t[:, None] - nx[None, :]) ** 2) / sigma_x**2)
    p_kernel = np.exp(-0.5 * ((p_t[:, None] - np_[None, :]) ** 2) / sigma_p**2)
    phi = (x_kernel[:, :, None] * p_kernel[:, None, :]).reshape(len(x_t), -1)

    mu = (phi * a[:, None]).mean(axis=0)
    xi = (phi.T @ phi) / len(x_t)
    coef = np.linalg.solve(xi + beta * np.eye(mu.size), mu)
    return float(mu @ coef / dt)


def estimate_sigma_curve(
    x: np.ndarray, p: np.ndarray, n: int, config: SimulationConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the entropy production curve for ``n`` trajectories."""

    dt = config.dt
    beta = n ** (-config.regularisation_exponent)
    nx, np_, sigma_x, sigma_p = construct_basis(x, p, config.basis_size)

    time_points = np.arange(0, config.total_steps + 1, config.sampling_interval)
    sigma_list = []

    for t in time_points:
        p_prev = p[:, t - 1] if t > 0 else None
        p_next = p[:, t + 1] if t < config.total_steps else None
        x_slice = x[:, [t, min(t + 1, config.total_steps)]]
        sigma_list.append(
            sigma_hat(x_slice, p[:, t], p_prev, p_next, beta, nx, np_, sigma_x, sigma_p, dt)
        )

    return time_points * dt, np.asarray(sigma_list)


def entropy_production_rate_theoretical(t: float) -> float:
    """Reference entropy production rate computed from the semi-analytical solution."""

    epsilon = 1e-5

    def sigma_total(time: float) -> float:
        k0 = 0.5 * (0.2**2 + 0.09)
        mean_p = 0.2 * math.exp(-time)
        var_p = 0.09 * math.exp(-2 * time) + (1 - math.exp(-2 * time))
        kt = 0.5 * (mean_p**2 + var_p)
        delta_s_env = k0 - kt

        var_x = 0.000625 + (1 - math.exp(-time)) ** 2 * 0.09 + (1 - math.exp(-2 * time))
        cov_xp = 1 - 1.91 * math.exp(-time) + 0.91 * math.exp(-2 * time)
        det_sigma = var_x * var_p - cov_xp**2
        det_sigma0 = 0.000625 * 0.09
        delta_s_sys = 0.5 * math.log(det_sigma / det_sigma0)
        return delta_s_env + delta_s_sys

    return (sigma_total(t + epsilon) - sigma_total(t - epsilon)) / (2 * epsilon)


def simulate_entropy_production(
    config: SimulationConfig, rng: np.random.Generator | None = None
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tuple[Tuple[str, np.ndarray], Tuple[str, np.ndarray]],
]:
    """Run the full simulation.

    Returns
    -------
    t_vals:
        Dense time grid used for the theoretical reference curve.
    sigma_theoretical:
        Entropy production rate evaluated on ``t_vals``.
    sample_times:
        Time grid associated with the Monte Carlo estimates.
    estimates:
        Tuple containing the estimated entropy production curves for the configured trajectory
        counts.
    """

    rng = np.random.default_rng() if rng is None else rng
    total_steps = config.total_steps

    x_small, p_small = simulate_underdamped(config.n_trajectories[0], config, rng)
    x_large, p_large = simulate_underdamped(config.n_trajectories[1], config, rng)

    times_small, sigma_small = estimate_sigma_curve(x_small, p_small, config.n_trajectories[0], config)
    times_large, sigma_large = estimate_sigma_curve(x_large, p_large, config.n_trajectories[1], config)

    if not np.array_equal(times_small, times_large):
        raise ValueError("Time grids of the two simulations do not match")

    t_vals = np.linspace(0, total_steps * config.dt, total_steps + 1)
    sigma_theoretical = np.asarray([entropy_production_rate_theoretical(t) for t in t_vals])

    estimates = (
        (f"N={config.n_trajectories[0]:,} trajectories", sigma_small),
        (f"N={config.n_trajectories[1]:,} trajectories", sigma_large),
    )

    return t_vals, sigma_theoretical, times_small, estimates


def main() -> None:
    config = SimulationConfig()
    t_vals, sigma_theoretical, sample_times, estimates = simulate_entropy_production(config)

    plt.figure(figsize=(6, 4))
    plt.plot(t_vals, sigma_theoretical, "k--", label="理论值")

    for label, sigma_estimate in estimates:
        marker = "o" if label.startswith("N=10,000") else "s"
        plt.plot(sample_times, sigma_estimate, marker=marker, label=label)

    plt.xlabel("时间 $t$")
    plt.ylabel("熵产生率 $\\sigma(t)$")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
