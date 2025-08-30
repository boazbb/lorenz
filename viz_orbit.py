#!/usr/bin/env python3
"""
Usage:
  python plot_orbit_poincare.py 123

What it does:
  - Loads /Users/makoto/code/lorenz_v2/data/DATA1/orbit{NUM}.dat
  - Plots the orbit in 3D
  - Draws the Poincaré plane z=27
  - Computes & plots intersection points of the orbit with z=27
  - Reads the symbol sequence for this orbit from DATA2/WINNERS,
    prints it to the console, and puts it in the figure title.
"""

import sys
from typing import List, Literal, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (ensures 3D projection is registered)
import os

# --- File paths (adjust if needed) ---
ORBIT_PATH = os.path.join("data", "DATA1")
SYMBOL_LENGTH_FILE = os.path.join("data", "DATA2", "WINNERS")

# --- I/O helpers -------------------------------------------------------------

def get_orbit_data(path: str) -> np.ndarray:
    """Read orbit file: 3 lines for x, y, z (space-separated). Returns (N, 3)."""
    with open(path, "r") as f:
        lines = f.readlines()
    assert len(lines) == 3, "file should have 3 lines"
    xs = lines[0].split()
    ys = lines[1].split()
    zs = lines[2].split()
    return np.stack([xs, ys, zs], axis=1).astype(float)

def get_symbol_sequence(num: int, winners_path: str = SYMBOL_LENGTH_FILE) -> str:
    """
    Parse a line like 'orbit 4449: AABBAB...' for the requested orbit number.
    Returns '' if not found.
    """
    try:
        with open(winners_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("orbit"):
                    continue
                # expected format: "orbit <num>: <SYMBOLS>"
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                lhs, rhs = parts[0].strip(), parts[1].strip()
                lhs_parts = lhs.split()
                if len(lhs_parts) != 2 or lhs_parts[0] != "orbit":
                    continue
                try:
                    file_num = int(lhs_parts[1])
                except ValueError:
                    continue
                if file_num == num:
                    return rhs
    except FileNotFoundError:
        pass
    return ""

# --- Poincaré intersections on z=27 ------------------------------------------

Direction = Literal["both", "up", "down"]

def _segment_crosses_plane(z0: float, z1: float, zsec: float, direction: Direction) -> bool:
    f0, f1 = z0 - zsec, z1 - zsec
    crosses = (f0 == 0.0) or (f1 == 0.0) or (f0 * f1 < 0.0)
    if not crosses:
        return False
    if direction == "both":
        return True
    going_up = (z1 > z0)
    return (direction == "up" and going_up) or (direction == "down" and not going_up)

def _linear_cross_point(p0: np.ndarray, p1: np.ndarray, zsec: float) -> np.ndarray:
    z0, z1 = p0[2], p1[2]
    dz = z1 - z0
    if dz == 0.0:
        return (p0 + p1) * 0.5
    u = (zsec - z0) / dz
    return p0 + u * (p1 - p0)

def intersections_from_samples(samples_xyz: np.ndarray,
                               z_section: float = 27.0,
                               direction: Direction = "up",
                               wrap: bool = True,
                               dedup_tol: float = 1e-10) -> np.ndarray:
    """
    Given (N,3) samples along one closed orbit, return all intersection points with z=z_section.
    """
    pts: List[np.ndarray] = []
    n = samples_xyz.shape[0]
    last = n if not wrap else n + 1
    for i in range(last - 1):
        p0 = samples_xyz[i % n]
        p1 = samples_xyz[(i + 1) % n]
        if _segment_crosses_plane(p0[2], p1[2], z_section, direction):
            p = _linear_cross_point(p0, p1, z_section)
            p[2] = z_section
            pts.append(p)
    if not pts:
        return np.empty((0, 3), dtype=float)
    # Deduplicate near-identical crossings
    out = [pts[0]]
    for p in pts[1:]:
        if np.min([np.linalg.norm(p - q) for q in out]) > dedup_tol:
            out.append(p)
    return np.vstack(out)

# --- Plotting -----------------------------------------------------------------

def plot_orbit_plane_and_hits(orbit_xyz: np.ndarray,
                              hits_xyz: np.ndarray,
                              orbit_num: int,
                              symbol_seq: str,
                              z_section: float = 27.0) -> None:
    # Figure & 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Orbit
    ax.plot(orbit_xyz[:, 0], orbit_xyz[:, 1], orbit_xyz[:, 2], lw=1.0, label="orbit")

    # Plane z = z_section (size set by orbit extents, with margin)
    xmin, xmax = np.min(orbit_xyz[:, 0]), np.max(orbit_xyz[:, 0])
    ymin, ymax = np.min(orbit_xyz[:, 1]), np.max(orbit_xyz[:, 1])
    dx, dy = xmax - xmin, ymax - ymin
    margin_x = 0.05 * (dx if dx > 0 else 1.0)
    margin_y = 0.05 * (dy if dy > 0 else 1.0)
    X = np.linspace(xmin - margin_x, xmax + margin_x, 20)
    Y = np.linspace(ymin - margin_y, ymax + margin_y, 20)
    XX, YY = np.meshgrid(X, Y)
    ZZ = np.full_like(XX, z_section)
    ax.plot_surface(XX, YY, ZZ, alpha=0.2, edgecolor="none")

    # Intersections
    if hits_xyz.size:
        ax.scatter(hits_xyz[:, 0], hits_xyz[:, 1], hits_xyz[:, 2],
                   s=30, depthshade=False, label="intersections", color="red")

    # Labels & title
    title = f"Orbit {orbit_num}"
    if symbol_seq:
        title += f" — symbols: {symbol_seq}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()

# --- Main ---------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_orbit_poincare.py <orbit_number>")
        sys.exit(2)

    num = int(sys.argv[1])
    orbit_path = os.path.join(ORBIT_PATH, f"orbit{num}.dat")
    orbit = get_orbit_data(orbit_path)

    # Compute intersections with z=27 (treat the data as one closed period)
    hits = intersections_from_samples(orbit, z_section=27.0, direction="up", wrap=True)

    # Load symbol sequence
    symbols = get_symbol_sequence(num)
    if symbols:
        print(f"Orbit {num} symbol sequence: {symbols}")
    else:
        print(f"Orbit {num}: symbol sequence not found in {SYMBOL_LENGTH_FILE}")

    # Plot
    plot_orbit_plane_and_hits(orbit, hits, num, symbols, z_section=27.0)

    print(f"Symbol sequence: {symbols}")
    print(f"Symbol sequence length: {len(symbols)}")
    print(f"Number of intersections: {hits.shape[0]}")

    assert hits.shape[0] == 2 * len(symbols), f"Number of intersections {hits.shape[0]} should be 2 * length of symbol sequence {len(symbols)}"

if __name__ == "__main__":
    main()
