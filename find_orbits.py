#!/usr/bin/env python3
"""
Robust Option A: Poincaré return map on z=27 with A/B itineraries (Lorenz, σ=10, ρ=28, β=8/3).
Find periodic orbits for all primitive words with lengths in [min_len, max_len] (default 2..4).

Fixes vs v1:
- Enforce a single crossing direction (decreasing z): only accept events with dz/dt < -dzdt_min.
- Require a minimum flight time between section hits (dt >= min_flight_time) to avoid grazing/tangencies.
- Remove strict rectangle gating for seeds; only the sign of x must match the first letter.
- Switch to scipy.optimize.least_squares for a more robust solve than root('hybr').
- Improved period computation and verification; skip length-1 words by default.

Usage:
    python find_lorenz_orbits_optionA.py --min_len 2 --max_len 4 --hits 10000 --seeds 1000 --verbose

Outputs:
    - Prints the found orbits
    - CSV: lorenz_orbits_len_[min]_[max].csv
"""

from __future__ import annotations
import math
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from tqdm import tqdm

# -------------------------- Lorenz system & section --------------------------

@dataclass
class LorenzParams:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0

@dataclass
class SectionConfig:
    z0: float = 27.0        # Poincaré plane z = z0
    # Optional loose ranges (not strictly enforced; kept for plotting or future use)
    y_range: Tuple[float, float] = (-60.0, 60.0)

def lorenz_rhs(t: float, X: np.ndarray, p: LorenzParams) -> np.ndarray:
    x, y, z = X
    return np.array([
        p.sigma * (y - x),
        -x * z + p.rho * x - y,
        x * y - p.beta * z
    ], dtype=float)

def label_from_x(x: float) -> str:
    return 'A' if x >= 0.0 else 'B'

# -------------------------- Integration helpers -----------------------------

@dataclass
class Hit:
    x: float
    y: float
    t: float        # absolute time of hit
    label: str      # 'A' if x>=0 else 'B'

class PoincareIntegrator:
    def __init__(self, params: LorenzParams, cfg: SectionConfig,
                 rtol: float = 1e-9, atol: float = 1e-12, max_step: float = 0.05,
                 direction: int = -1, dzdt_min: float = 0.1, min_flight_time: float = 0.25):
        """
        direction: -1 for decreasing z crossings only; +1 for increasing; 0 for both (not recommended).
        dzdt_min: minimum |dz/dt| required at a valid crossing (transversality guard).
        min_flight_time: minimum time between accepted crossings.
        """
        assert direction in (-1, 0, 1)
        self.p = params
        self.cfg = cfg
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.direction = direction
        self.dzdt_min = dzdt_min
        self.min_flight_time = min_flight_time

    def _nudge_along_flow(self, X: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        f = lorenz_rhs(0.0, X, self.p)
        fn = np.linalg.norm(f)
        if not np.isfinite(fn) or fn == 0.0:
            return X
        return X + (eps / fn) * f

    def _event_func(self, z0: float):
        def ev(t, X):
            return X[2] - z0
        ev.terminal = True
        # direction: 1 means ev increasing through 0; -1 decreasing; 0 both
        ev.direction = float(self.direction)
        return ev

    def next_hit(self, X0: np.ndarray, t0: float, tmax: float = 12.0, max_tries: int = 20) -> Hit:
        """
        Integrate forward to the next *valid* crossing of z = z0 with the configured direction,
        transversality, and minimum flight time. Returns the Hit.
        """
        z0 = self.cfg.z0
        # ensure we're not exactly on the section
        if abs(X0[2] - z0) < 1e-12:
            X0 = self._nudge_along_flow(X0)
        t_start = float(t0)
        X_start = X0.astype(float).copy()

        # We'll allow several attempts to skip invalid/grazing hits.
        for attempt in range(max_tries):
            ev = self._event_func(z0)
            # integrate up to t_start + tmax (per attempt)
            sol = solve_ivp(
                fun=lambda t, X: lorenz_rhs(t, X, self.p),
                t_span=(t_start, t_start + tmax),
                y0=X_start,
                events=ev,
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )
            if sol.status == 1 and len(sol.t_events[0]) > 0:
                t_hit = float(sol.t_events[0][0])
                X_hit = sol.y_events[0][0].astype(float)
                xh, yh, zh = X_hit
                dt = t_hit - t_start
                dzdt = lorenz_rhs(t_hit, X_hit, self.p)[2]
                # Direction check per config
                dir_ok = True
                if self.direction == -1:
                    dir_ok = dzdt < -self.dzdt_min
                elif self.direction == +1:
                    dir_ok = dzdt > self.dzdt_min
                # Minimum flight time check
                dt_ok = dt >= self.min_flight_time
                if dir_ok and dt_ok:
                    lab = label_from_x(xh)
                    return Hit(float(xh), float(yh), t_hit, lab)
                # otherwise, continue a bit past the hit and try again
                X_start = self._nudge_along_flow(np.array([xh, yh, zh]))
                t_start = t_hit + 1e-7
                continue
            else:
                # failed this attempt; extend time window slightly and try again
                t_start = t_start + tmax * 0.5
                X_start = self._nudge_along_flow(X_start)

        raise RuntimeError("Failed to find next valid Poincaré hit within attempts/time window.")

    def integrate_k_hits(self, u0_xy: Tuple[float, float], k: int,
                         start_time: float = 0.0, per_hit_tmax: float = 12.0) -> Tuple[List[Hit], Tuple[float,float]]:
        x0, y0 = u0_xy
        X = np.array([x0, y0, self.cfg.z0], dtype=float)
        t = start_time
        hits: List[Hit] = []
        for _ in range(k):
            h = self.next_hit(X, t, tmax=per_hit_tmax)
            hits.append(h)
            X = np.array([h.x, h.y, self.cfg.z0], dtype=float)
            t = h.t
            X = self._nudge_along_flow(X)
        return hits, (X[0], X[1])

# ------------------------------ Words utils ---------------------------------

def is_primitive(word: str) -> bool:
    return all(word != word[:d] * (len(word)//d) for d in range(1, len(word)) if len(word) % d == 0)

def rotations(word: str) -> List[str]:
    return [word[i:] + word[:i] for i in range(len(word))]

def canonical_rotation(word: str) -> str:
    rots = rotations(word)
    return min(rots)

def generate_words(min_len: int = 2, max_len: int = 4, primitive_only: bool = True, dedupe_rotations: bool = True) -> List[str]:
    words: Set[str] = set()
    for L in range(min_len, max_len + 1):
        for tup in itertools.product('AB', repeat=L):
            w = ''.join(tup)
            if primitive_only and not is_primitive(w):
                continue
            if dedupe_rotations:
                words.add(canonical_rotation(w))
            else:
                words.add(w)
    # Sort: by length then lexicographic
    return sorted(words, key=lambda s: (len(s), s))

# --------------------------- Seeds from simulation ---------------------------

def collect_hits(integrator: PoincareIntegrator, n_hits: int = 3000,
                 X0: np.ndarray = np.array([0.0, 1.0, 1.05]), tmax_per_hit: float = 12.0) -> List[Hit]:
    hits: List[Hit] = []
    X = X0.astype(float).copy()
    t = 0.0
    try:
        for _ in tqdm(range(n_hits)):
            h = integrator.next_hit(X, t, tmax=tmax_per_hit)
            hits.append(h)
            X = np.array([h.x, h.y, integrator.cfg.z0], dtype=float)
            t = h.t
            X = integrator._nudge_along_flow(X)
    except RuntimeError:
        pass
    return hits

def find_seeds_for_word(hits: List[Hit], word: str, max_seeds: int = 100) -> List[Tuple[float,float]]:
    k = len(word)
    seeds: List[Tuple[float,float]] = []
    labels = ''.join(h.label for h in hits)
    # scan contiguous subsequences of length k matching the word
    for i in range(len(labels) - k):
        if labels[i:i+k] == word:
            seeds.append((hits[i].x, hits[i].y))
            if len(seeds) >= max_seeds:
                break
    return seeds

# --------------------------- Fixed point solving -----------------------------

@dataclass
class OrbitResult:
    word: str
    canonical_word: str
    x: float
    y: float
    period: float
    k: int
    labels: str
    success: bool
    message: str

def solve_for_word(integrator: PoincareIntegrator, word: str, seeds: List[Tuple[float,float]],
                   per_hit_tmax: float = 12.0, verbose: bool = False) -> Optional[OrbitResult]:
    k = len(word)

    def Pk(u: np.ndarray) -> Tuple[np.ndarray, str, float]:
        hits, (xk, yk) = integrator.integrate_k_hits((u[0], u[1]), k, per_hit_tmax=per_hit_tmax)
        labels = ''.join(h.label for h in hits)
        # accurate period as difference of absolute times
        T = hits[-1].t - hits[0].t if k > 1 else (hits[0].t)
        return np.array([xk, yk], dtype=float), labels, float(T)

    def F(u: np.ndarray) -> np.ndarray:
        try:
            xk, labels, _T = Pk(u)
            return xk - u
        except Exception:
            return np.array([1e3, 1e3], dtype=float)

    for seed in seeds:
        # only require that the sign matches the first letter
        if (seed[0] >= 0.0 and word[0] != 'A') or (seed[0] < 0.0 and word[0] != 'B'):
            continue
        u0 = np.array([seed[0], seed[1]], dtype=float)

        try:
            sol = least_squares(lambda u: F(u), u0, method='trf', xtol=1e-10, ftol=1e-12, gtol=1e-12, max_nfev=300)
        except Exception as e:
            if verbose:
                print(f"[{word}] seed exception: {e}")
            continue

        if not sol.success:
            if verbose:
                print(f"[{word}] LSQ did not converge from seed {seed}: {sol.message}")
            continue

        u = sol.x
        try:
            xk, labels, T = Pk(u)
        except Exception as e:
            if verbose:
                print(f"[{word}] verification failed: {e}")
            continue

        # Check itinerary and sanity
        if labels != word:
            if verbose:
                print(f"[{word}] converged to wrong labels {labels} from seed {seed}. Skipping.")
            continue

        # Sanity: enforce a reasonable minimum total time
        if T < k * integrator.min_flight_time * 0.9:
            if verbose:
                print(f"[{word}] suspiciously short period {T:.4g}; skipping.")
            continue

        if verbose:
            print(f"[{word}] success at u=({u[0]:.6f},{u[1]:.6f}), period≈{T:.6f}, labels={labels}")
        return OrbitResult(
            word=word,
            canonical_word=canonical_rotation(word),
            x=float(u[0]),
            y=float(u[1]),
            period=float(T),
            k=k,
            labels=labels,
            success=True,
            message="ok",
        )

    return None

# --------------------------- Main orchestration ------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Find Lorenz periodic orbits (Option A) for itineraries within a given length range.")
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=4)
    parser.add_argument("--hits", type=int, default=5000, help="number of Poincaré hits to collect for seeding")
    parser.add_argument("--seeds", type=int, default=200, help="max seeds per word")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--direction", type=int, default=-1, choices=[-1,0,1], help="-1: dz/dt<0 only (default), 1: dz/dt>0 only, 0: both")
    parser.add_argument("--dzdt_min", type=float, default=0.10, help="min |dz/dt| at a valid crossing")
    parser.add_argument("--min_flight_time", type=float, default=0.25, help="min time between successive hits")
    parser.add_argument("--per_hit_tmax", type=float, default=12.0, help="max time window to find each next hit")
    parser.add_argument("--primitive_only", action="store_true", default=True)
    parser.add_argument("--all-words", action="store_true", help="include non-primitive and duplicate rotations")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_hits", type=str, default="", help="optional path to save collected hits as npz")
    parser.add_argument("--load_hits", type=str, default="", help="optional path to load collected hits from npz")
    parser.add_argument("--try_double", action="store_true", help="try the double of word")
    args = parser.parse_args()

    params = LorenzParams()
    cfg = SectionConfig()
    integrator = PoincareIntegrator(params, cfg, rtol=args.rtol, atol=args.atol,
                                    direction=args.direction, dzdt_min=args.dzdt_min,
                                    min_flight_time=args.min_flight_time)



    if args.load_hits:
        print(f"Loading hits from {args.load_hits}...")
        data = np.load(args.load_hits)
        hits = [Hit(x=float(x), y=float(y), t=float(t), label=str(lab)) for x,y,t,lab in zip(data['x'], data['y'], data['t'], data['label'])]
    else:
        print(f"Collecting {args.hits} section hits for seeds (direction={args.direction}, dzdt_min={args.dzdt_min}, min_dt={args.min_flight_time})...")
        hits = collect_hits(integrator, n_hits=args.hits, tmax_per_hit=args.per_hit_tmax)
        print(f"Collected {len(hits)} hits.")
        if args.save_hits:
            print(f"Saving hits to {args.save_hits}...")
            np.savez(args.save_hits,
                     x=np.array([h.x for h in hits], dtype=float),
                     y=np.array([h.y for h in hits], dtype=float),
                     t=np.array([h.t for h in hits], dtype=float),
                     label=np.array([h.label for h in hits], dtype='U1'))
            print("Saved.")

    primitive_only = not args.all_words
    words = generate_words(args.min_len, args.max_len, primitive_only=primitive_only, dedupe_rotations=True)

    print(f"Trying words (length {args.min_len}..{args.max_len}): {', '.join(words)}")
    results: List[OrbitResult] = []
    seen_keys: Set[Tuple[str, int, int]] = set()

    for w in tqdm(words):
        if args.try_double:
            seeds = find_seeds_for_word(hits, w*3, max_seeds=args.seeds)
            if not seeds:
                print(f"[{w}] no seeds found for double word; trying single.")
                seeds = find_seeds_for_word(hits, w, max_seeds=args.seeds)
        else:
            seeds = find_seeds_for_word(hits, w, max_seeds=args.seeds)
        if not seeds:
            if args.verbose:
                print(f"[{w}] no seeds found in the chaotic run; skipping")
            continue
        res = solve_for_word(integrator, w, seeds, per_hit_tmax=args.per_hit_tmax, verbose=args.verbose)
        if res is None:
            if args.verbose:
                print(f"[{w}] no orbit found (failed to converge or wrong itinerary).")
            continue

        # dedupe across words that are rotations of each other (store under canonical)
        key = (res.canonical_word, int(round(res.x * 1e6)), int(round(res.y * 1e6)))
        if key in seen_keys:
            if args.verbose:
                print(f"[{w}] duplicate (by rotation); skipping record.")
            continue

        seen_keys.add(key)
        results.append(res)

    if not results:
        print("No periodic orbits found. Try increasing --hits or loosening thresholds (e.g., --dzdt_min 0.05, --min_flight_time 0.15).")
        return

    # Pretty print
    print("\n=== Found periodic orbits ===")
    print(f"{'word':<6} {'k':>2} {'period':>12} {'x(z=27)':>14} {'y(z=27)':>14} {'labels':>8}")
    for r in sorted(results, key=lambda R: (R.k, R.word)):
        print(f"{r.word:<6} {r.k:>2} {r.period:12.6f} {r.x:14.6f} {r.y:14.6f} {r.labels:>8}")

    # Also dump a CSV
    import csv
    csv_path = "lorenz_orbits_len_{}_{}.csv".format(args.min_len, args.max_len)
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["word", "k", "canonical_word", "period", "x_at_z27", "y_at_z27", "labels"])
        for r in results:
            wtr.writerow([r.word, r.k, r.canonical_word, r.period, r.x, r.y, r.labels])
    print(f"\nSaved CSV: {csv_path}")

if __name__ == "__main__":
    main()