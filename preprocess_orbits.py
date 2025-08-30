#!/usr/bin/env python3
"""
Builds a dictionary: Dict[int, List[PeriodicOrbit]], keyed by symbol sequence length.
Each PeriodicOrbit holds: symbol_sequence, length, points (Nx3), stable_characteristic_multiplier.

Saves to: periodic_orbits_by_length.npz  (object array; load with allow_pickle=True)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, is_dataclass
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
import os
import json

ORBIT_PATH = os.path.join("data", "DATA1", "orbit{NUM}.dat")
PERIOD_PATH = os.path.join("data", "DATA1", "T{NUM}.dat")

SYMBOL_LENGTH_FILE = os.path.join("data", "DATA2", "WINNERS")
# MAX_NUM = 1375
MAX_NUM = 100 - 31

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

@dataclass
class PeriodicOrbit:
    symbol_sequence: str
    length: int
    stable_characteristic_multiplier: float

# ---------------- I/O helpers -------------------------------------------------
def get_orbit_data(path: str) -> np.ndarray:
    with open(path, "r") as f:
        lines = f.readlines()
    assert len(lines) == 3, "file should have 3 lines"
    xs = lines[0].split()
    ys = lines[1].split()
    zs = lines[2].split()
    return np.stack([xs, ys, zs], axis=1).astype(float)

def get_period(path: str) -> float:
    with open(path, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1, "file should have 1 line"
    return float(lines[0])

def parse_symbol_length_file(path: str):
    """Yield (orbit_num, symbol_length, symbol_sequence) for entries up to MAX_NUM."""
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("orbit"):
                continue
            num_part, seq_part = line.split(":", 1)
            try:
                orbit_num = int(num_part.split()[1])
            except Exception:
                continue
            if orbit_num > MAX_NUM:
                break
            symbol_sequence = seq_part.strip()
            yield orbit_num, len(symbol_sequence), symbol_sequence

# ---------------- Lorenz + variational system --------------------------------
def lorenz_system_and_variational(t, Y):
    """
    12D system: (x,y,z) + flattened Phi (3x3) with dPhi/dt = J(x,y,z) @ Phi.
    """
    x, y, z = Y[:3]
    Phi = Y[3:].reshape((3, 3))

    # Lorenz
    dxdt = SIGMA * (y - x)
    dydt = x * (RHO - z) - y
    dzdt = x * y - BETA * z

    # Jacobian
    J = np.array([
        [-SIGMA, SIGMA, 0.0],
        [RHO - z, -1.0, -x],
        [y, x, -BETA]
    ])

    dPhidt = J @ Phi
    return np.concatenate(([dxdt, dydt, dzdt], dPhidt.flatten()))

def compute_monodromy_eigs(orbit_num: int) -> np.ndarray:
    """
    Integrate one period starting from the first orbit point, carrying the variational
    equations; return Floquet multipliers (eigenvalues of monodromy).
    """
    orbit = get_orbit_data(ORBIT_PATH.format(NUM=orbit_num))
    T = get_period(PERIOD_PATH.format(NUM=orbit_num))
    x0 = orbit[0]
    Phi0 = np.eye(3).flatten()
    Y0 = np.concatenate([x0, Phi0])
    sol = solve_ivp(
        lorenz_system_and_variational,
        t_span=[0.0, T],
        y0=Y0,
        t_eval=[T],
        rtol=1e-9,
        atol=1e-12,
        dense_output=False,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for orbit {orbit_num}: {sol.message}")
    YT = sol.y[:, -1]
    M = YT[3:].reshape(3, 3)
    return np.linalg.eigvals(M)

def extract_stable_multiplier(eigs: np.ndarray) -> float:
    """
    Return |λ_s| for the *stable* nontrivial multiplier:
      - discard the eigenvalue nearest 1 (the trivial flow multiplier),
      - among the remaining two, choose the one with |λ| < 1 if present,
        otherwise choose the one with the smaller |λ| (fallback).
    """
    eigs = np.asarray(eigs, dtype=complex)
    # remove the one closest to +1
    i_trivial = int(np.argmin(np.abs(eigs - 1.0)))
    nontrivial = np.delete(eigs, i_trivial)
    mags = np.abs(nontrivial)
    # prefer the one < 1
    candidates = nontrivial[mags < 1.0]
    if candidates.size > 0:
        return float(np.abs(candidates[np.argmin(np.abs(candidates))]))
    # fallback: smaller magnitude among the two
    return float(np.min(np.abs(nontrivial)))

# ---------------- Main assembly ----------------------------------------------
def build_and_dump(output_path: str = "periodic_orbits_by_length.json") -> Dict[int, List[PeriodicOrbit]]:
    # Preload symbol sequences (restrict to <= MAX_NUM)
    entries = list(parse_symbol_length_file(SYMBOL_LENGTH_FILE))
    by_num = {n: (L, s) for (n, L, s) in entries}

    orbits_by_length: Dict[int, List[PeriodicOrbit]] = defaultdict(list)

    # Iterate over all orbits mentioned in WINNERS (safer than raw range)
    for orbit_num, L, sym in tqdm(entries, total=len(entries), desc="Processing orbits"):
        orbit_path = ORBIT_PATH.format(NUM=orbit_num)
        if not os.path.exists(orbit_path):
            # Skip cleanly if the file is missing
            assert False, f"orbit file {orbit_path} does not exist"

        # Load points
        pts = get_orbit_data(orbit_path)

        # Compute multipliers, pick stable one (magnitude)
        try:
            eigs = compute_monodromy_eigs(orbit_num)
            stable_mag = extract_stable_multiplier(eigs)
        except Exception as e:
            # If something goes wrong, store NaN and continue
            stable_mag = float("nan")

        po = PeriodicOrbit(
            symbol_sequence=sym,
            length=L,
            stable_characteristic_multiplier=stable_mag,
        )
        orbits_by_length[L].append(po)

    print(f"Found {sum(len(orbits) for orbits in orbits_by_length.values())} periodic orbits")
    for m in sorted(orbits_by_length.keys()):
        print(f"m={m}: {len(orbits_by_length[m])}")
    
    print(f"Saving to {output_path}")
    with open(output_path, "w") as f:
        json.dump(dict(orbits_by_length), f, default=json_default, indent=2)

    return orbits_by_length

def json_default(o):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, np.ndarray):       # in case you add `points` back
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


if __name__ == "__main__":
    build_and_dump()
