import argparse
import math
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_experiment(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load experimental strain-stress data from CSV or Excel.
    Expected columns (case-insensitive, spaces tolerated): 'strain', 'stress'.
    Returns (strain, stress) as float64 numpy arrays sorted by strain ascending.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        try:
            data = pd.read_excel(file_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to read Excel file. Ensure 'openpyxl' is installed or supply a CSV."
            ) from exc
    elif ext in (".csv", ".txt"):
        data = pd.read_csv(file_path)
    else:
        # Try CSV as a fallback
        data = pd.read_csv(file_path)

    data = data.rename(columns=lambda x: str(x).strip().lower().replace(" ", "_"))

    # Accept common variants
    strain_col = None
    stress_col = None
    for c in data.columns:
        if strain_col is None and ("strain" in c or c == "ε" or c == "epsilon"):
            strain_col = c
        if stress_col is None and ("stress" in c or c in {"q", "deviatoric_stress"}):
            stress_col = c

    if strain_col is None or stress_col is None:
        raise ValueError(
            f"Could not find 'strain' and 'stress' columns. Columns present: {list(data.columns)}"
        )

    strain = np.asarray(data[strain_col].values, dtype=np.float64)
    stress = np.asarray(data[stress_col].values, dtype=np.float64)

    # Drop NaNs and sort by strain
    mask = np.isfinite(strain) & np.isfinite(stress)
    strain, stress = strain[mask], stress[mask]
    order = np.argsort(strain)
    strain = strain[order]
    stress = stress[order]

    return strain, stress


def hypopwa(T: torch.Tensor,
            D: torch.Tensor,
            c1: float,
            c2: float,
            c3: float,
            c4: float,
            fd: float = 1.0) -> torch.Tensor:
    """
    Hypoplastic stress rate: Tdot = c1*tr(T)*D + c2*tr(D)*T + c3*(tr(TD)/tr(T))*T + fd*c4*(T + Tdev)*||D||_F
    Uses robust guards for small tr(T) to avoid division by zero.
    T, D are 3x3 tensors.
    """
    dtype = T.dtype
    device = T.device

    trT = torch.trace(T)
    trD = torch.trace(D)
    trT_over_3 = trT / 3.0
    trTD = torch.sum(T * D)
    DD = torch.linalg.norm(D)

    I = torch.eye(3, dtype=dtype, device=device)
    Tdev = T - trT_over_3 * I

    Tk1 = c1 * trT * D
    Tk2 = c2 * trD * T

    eps = torch.tensor(1e-12, dtype=dtype, device=device)
    trT_safe = torch.where(torch.abs(trT) > eps, trT, eps * torch.sign(trT + eps))
    Tk3 = c3 * (trTD / trT_safe) * T

    Tk4 = fd * c4 * (T + Tdev) * DD
    return Tk1 + Tk2 + Tk3 + Tk4


@torch.no_grad()
def newton_solve_D22(Tn: torch.Tensor,
                     d11: float,
                     c: np.ndarray,
                     fd: float = 1.0,
                     D22_init: Optional[float] = None,
                     tol: float = 1e-6,
                     max_iter: int = 30,
                     damping: float = 0.5,
                     h: float = 1e-6) -> Tuple[float, bool]:
    """
    Solve for D22 such that Tdot[2,2] = 0 (keep lateral stress rate zero).
    Returns (D22, converged).
    """
    device = Tn.device
    dtype = Tn.dtype
    c1, c2, c3, c4 = [float(x) for x in c]
    if D22_init is None:
        D22 = -0.5 * d11
    else:
        D22 = float(D22_init)

    for _ in range(max_iter):
        D = torch.tensor([[d11, 0.0, 0.0],
                          [0.0, D22, 0.0],
                          [0.0, 0.0, D22]], dtype=dtype, device=device)
        Tdot = hypopwa(Tn, D, c1, c2, c3, c4, fd)
        f = Tdot[2, 2]
        if torch.isfinite(f) and torch.abs(f) <= tol:
            return float(D22), True

        Dp = torch.tensor([[d11, 0.0, 0.0],
                           [0.0, D22 + h, 0.0],
                           [0.0, 0.0, D22 + h]], dtype=dtype, device=device)
        Tdot_p = hypopwa(Tn, Dp, c1, c2, c3, c4, fd)
        df = (Tdot_p[2, 2] - f) / h
        if not torch.isfinite(df) or torch.abs(df) < 1e-16:
            df = torch.tensor(1e-16, dtype=dtype, device=device)

        step = (f / df).item()
        step = float(np.clip(step, -1.0, 1.0))
        D22 = D22 - damping * step

    return float(D22), False


@torch.no_grad()
def simulate_curve(c_params: np.ndarray,
                   experimental_strain: np.ndarray,
                   sigma3_kpa: float,
                   fd: float = 1.0,
                   compression_positive: bool = True,
                   device: Optional[torch.device] = None,
                   tol: float = 1e-6,
                   max_iter: int = 30) -> Optional[np.ndarray]:
    """
    Simulate deviatoric stress q over the experimental strain grid using the hypoplastic model.
    - Confining pressure is held constant via Tdot[22]=0.
    - Initial state is isotropic with T = diag(-sigma3, -sigma3, -sigma3) (compression negative convention).
    Returns q_pred aligned with experimental_strain, or None if simulation fails.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # Sign convention: compression negative stresses.
    sigma3 = float(abs(sigma3_kpa))
    Tn = torch.diag(torch.tensor([-sigma3, -sigma3, -sigma3], dtype=dtype, device=device))

    c = np.asarray(c_params, dtype=np.float64)

    strain = np.asarray(experimental_strain, dtype=np.float64)
    if strain.ndim != 1 or strain.size < 2:
        return None

    # Convert experimental increments into D11 per step; keep dt=1 so D == DeltaE.
    d_eps = np.diff(strain)
    # If experimental compression is positive, map to negative D11 to match model convention.
    if compression_positive:
        d11_steps = -np.abs(d_eps)
    else:
        d11_steps = np.asarray(d_eps, dtype=np.float64)

    q_pred = np.zeros_like(strain)
    q_pred[0] = (Tn[1, 1] - Tn[0, 0]).item()

    D22_guess: Optional[float] = None
    for i, d11 in enumerate(d11_steps, start=1):
        D22, ok = newton_solve_D22(Tn, float(d11), c, fd=fd, D22_init=D22_guess,
                                   tol=tol, max_iter=max_iter)
        if not ok:
            return None

        D = torch.tensor([[d11, 0.0, 0.0],
                          [0.0, D22, 0.0],
                          [0.0, 0.0, D22]], dtype=dtype, device=device)
        Tdot = hypopwa(Tn, D, float(c[0]), float(c[1]), float(c[2]), float(c[3]), fd)

        if not torch.all(torch.isfinite(Tdot)):
            return None

        # dt = 1 per increment (so D equals DeltaE per step)
        Tn = Tn + Tdot
        D22_guess = D22

        q_pred[i] = (Tn[1, 1] - Tn[0, 0]).item()

        if not torch.all(torch.isfinite(Tn)):
            return None

    return q_pred


def mse(a: np.ndarray, b: np.ndarray) -> float:
    err = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(err * err))


def tournament_selection(pop: np.ndarray, fit: np.ndarray, num: int, k: int = 3) -> np.ndarray:
    sel = []
    n = len(pop)
    for _ in range(num):
        idx = np.random.choice(n, size=min(k, n), replace=False)
        best = idx[np.argmin(fit[idx])]
        sel.append(pop[best])
    return np.asarray(sel)


def crossover(parents: np.ndarray, offspring_size: Tuple[int, int]) -> np.ndarray:
    off = np.empty(offspring_size, dtype=np.float64)
    dim = offspring_size[1]
    for i in range(offspring_size[0]):
        i1 = np.random.randint(0, parents.shape[0])
        i2 = np.random.randint(0, parents.shape[0])
        alpha = 2 * np.random.rand(dim) - 0.5  # [-0.5, 1.5]
        rand_factor = np.random.rand(dim) * 0.3
        off[i] = alpha * parents[i1] + (1.0 - alpha) * parents[i2] + rand_factor * (parents[i1] - parents[i2])
    return off


def mutation(offspring: np.ndarray, ranges: np.ndarray, mutation_rate: float = 0.2) -> np.ndarray:
    dim = offspring.shape[1]
    scale = ranges[:, 1] - ranges[:, 0]
    for i in range(offspring.shape[0]):
        if np.random.rand() < mutation_rate:
            if np.random.rand() < 0.5:
                delta = np.random.standard_cauchy(dim) * 0.5
            else:
                delta = np.random.normal(0.0, 0.8, dim)
            offspring[i] += 3.0 * delta * scale
            # Reflect at bounds
            below = offspring[i] < ranges[:, 0]
            above = offspring[i] > ranges[:, 1]
            offspring[i][below] = 2 * ranges[below, 0] - offspring[i][below]
            offspring[i][above] = 2 * ranges[above, 1] - offspring[i][above]
            # Clip
            offspring[i] = np.clip(offspring[i], ranges[:, 0], ranges[:, 1])
    return offspring


def ga_calibrate(experimental_strain: np.ndarray,
                 experimental_stress: np.ndarray,
                 sigma3_kpa: float,
                 param_ranges: np.ndarray,
                 pop_size: int = 200,
                 max_gen: int = 80,
                 elite_frac: float = 0.05,
                 mutation_rate: float = 0.2,
                 seed: Optional[int] = 42,
                 patience: int = 15) -> Tuple[np.ndarray, float, list]:
    if seed is not None:
        np.random.seed(seed)

    dim = param_ranges.shape[0]
    pop = np.random.uniform(low=param_ranges[:, 0], high=param_ranges[:, 1], size=(pop_size, dim))
    best_fit = math.inf
    best_params = None
    history = []
    no_improve = 0

    elite_num = max(1, int(round(elite_frac * pop_size)))

    for _ in tqdm(range(max_gen), desc="GA", ncols=80):
        fits = np.full(pop.shape[0], np.nan, dtype=np.float64)
        for i in range(pop.shape[0]):
            q_pred = simulate_curve(pop[i], experimental_strain, sigma3_kpa)
            if q_pred is None or not np.all(np.isfinite(q_pred)) or len(q_pred) != len(experimental_stress):
                fits[i] = 1e9
                continue
            fits[i] = mse(q_pred, experimental_stress)

        # Track best
        gen_best_idx = int(np.argmin(fits))
        gen_best_fit = float(fits[gen_best_idx])
        gen_best_params = pop[gen_best_idx].copy()
        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best_params = gen_best_params
            no_improve = 0
        else:
            no_improve += 1
        history.append(best_fit)

        if no_improve >= patience:
            break

        # Elitism
        elite_idx = np.argsort(fits)[:elite_num]
        elites = pop[elite_idx]

        # Selection, crossover, mutation
        parents = tournament_selection(pop, fits, pop_size - elite_num, k=3)
        offspring = crossover(parents, (pop_size - elite_num, dim))
        offspring = mutation(offspring, param_ranges, mutation_rate=mutation_rate)

        pop = np.vstack([elites, offspring])

    assert best_params is not None
    return best_params, best_fit, history


def default_param_ranges() -> np.ndarray:
    # Default ranges (c1..c4) guided by typical values in the user's script
    # c1<0, c2>0, c3<0, c4<0
    return np.array([
        [-200.0, -100.0],   # c1
        [300.0, 600.0],     # c2
        [-2000.0, -500.0],  # c3
        [-60.0, -20.0],     # c4
    ], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate hypoplastic model parameters (c1..c4) using GA.")
    p.add_argument("--data", required=True, help="Path to CSV/Excel with columns: strain, stress")
    p.add_argument("--sigma3", type=float, required=True, help="Confining pressure (kPa), positive value")
    p.add_argument("--pop", type=int, default=200, help="GA population size")
    p.add_argument("--gen", type=int, default=80, help="GA max generations")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--patience", type=int, default=15, help="Early-stop patience (generations without improvement)")
    p.add_argument("--savefig", default="calibration_result.png", help="Path to save the result plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    strain, stress = load_experiment(args.data)

    # Optional pre-rounding removed for fidelity; ensure float64
    strain = np.asarray(strain, dtype=np.float64)
    stress = np.asarray(stress, dtype=np.float64)

    # Run GA
    ranges = default_param_ranges()
    best_params, best_fit, hist = ga_calibrate(
        experimental_strain=strain,
        experimental_stress=stress,
        sigma3_kpa=args.sigma3,
        param_ranges=ranges,
        pop_size=args.pop,
        max_gen=args.gen,
        seed=args.seed,
        patience=args.patience,
    )

    # Re-simulate with best params
    q_best = simulate_curve(best_params, strain, sigma3_kpa=args.sigma3)

    print("Best parameters (c1..c4):", np.round(best_params, 6))
    print("Best MSE:", best_fit)

    # Plot convergence
    plt.figure(figsize=(10, 4))
    plt.plot(hist, "b-", lw=2)
    plt.xlabel("Generation")
    plt.ylabel("Best MSE")
    plt.title("GA Convergence")
    plt.grid(True)

    # Plot fit
    plt.figure(figsize=(10, 5))
    plt.plot(strain, stress, "ro", label="Experimental")
    if q_best is not None:
        plt.plot(strain, q_best, "b-", label="Model")
    plt.xlabel("Axial strain ε1")
    plt.ylabel("Deviatoric stress q (kPa)")
    plt.title("Hypoplastic Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.savefig, dpi=150)
    print(f"Saved plot to: {args.savefig}")


if __name__ == "__main__":
    main()

