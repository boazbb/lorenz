#!/usr/bin/env python3

from __future__ import annotations
import os
import sys
import argparse
from typing import Dict, List, Set, Tuple

# -------------------------- Utilities --------------------------

def rotations(word: str) -> List[str]:
    return [word[i:] + word[:i] for i in range(len(word))]

def canonical_rotation(word: str) -> str:
    return min(rotations(word)) if word else word

def parse_winners(path: str) -> Dict[int, Set[str]]:
    winners_by_len: Dict[int, Set[str]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # Orbit listing lines
                if "orbit" in line and ":" in line:
                    try:
                        seq = line.split(":", 1)[1].strip()
                    except Exception:
                        continue
                    if not seq or any(c not in "AB" for c in seq):
                        continue
                    L = len(seq)
                    winners_by_len.setdefault(L, set()).add(canonical_rotation(seq))
    except FileNotFoundError:
        raise
    return winners_by_len

def read_csv_words(csv_path: str) -> List[Tuple[str,int]]:
    import csv
    rows: List[Tuple[str,int]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if "word" not in rdr.fieldnames:
            raise ValueError(f'"word" column not found in CSV header: {rdr.fieldnames}')
        # k is optional; if missing compute len(word)
        k_present = "k" in rdr.fieldnames
        for row in rdr:
            w = row["word"].strip()
            if not w:
                continue
            k = int(row["k"]) if k_present and row.get("k","").strip() else len(w)
            rows.append((w, k))
    return rows

def default_winners_path(cli_path: str | None) -> str:
    if cli_path:
        return cli_path
    env_p = os.getenv("WINNERS_PATH")
    if env_p and os.path.exists(env_p):
        return env_p
    here = os.path.join(os.path.dirname(__file__ or "."), "WINNERS")
    if os.path.exists(here):
        return here
    legacy = os.path.join("data", "DATA2", "WINNERS")
    return legacy

# -------------------------- Main --------------------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate found Lorenz A/B symbolic orbits against a WINNERS list.")
    ap.add_argument("csv", help="Path to CSV with found orbits (must include 'word' column)")
    ap.add_argument("--winners", help="Path to WINNERS file; if omitted, tries $WINNERS_PATH, then ./WINNERS, then data/DATA2/WINNERS")
    ap.add_argument("--min_len", type=int, default=None, help="Minimum word length to validate (default: min length in CSV)")
    ap.add_argument("--max_len", type=int, default=None, help="Maximum word length to validate (default: max length in CSV)")
    ap.add_argument("--ignore_mismatch", action="store_true", help="Exit with code 0 even if mismatches are found")
    args = ap.parse_args(argv)

    winners_path = default_winners_path(args.winners)
    if not os.path.exists(winners_path):
        print(f"ERROR: WINNERS file not found at: {winners_path}", file=sys.stderr)
        return 2

    # Load inputs
    try:
        csv_rows = read_csv_words(args.csv)
    except Exception as e:
        print(f"ERROR reading CSV: {e}", file=sys.stderr)
        return 2

    try:
        winners_by_len = parse_winners(winners_path)
    except Exception as e:
        print(f"ERROR reading WINNERS file: {e}", file=sys.stderr)
        return 2

    if not csv_rows:
        print("No rows found in CSV.", file=sys.stderr)
        return 2

    # Decide length range
    csv_lens = [k for _, k in csv_rows]
    Lmin = args.min_len if args.min_len is not None else min(csv_lens)
    Lmax = args.max_len if args.max_len is not None else max(csv_lens)
    if Lmin > Lmax:
        Lmin, Lmax = Lmax, Lmin

    # Canonicalize CSV words and bucket by length
    by_len_csv: Dict[int, Set[str]] = {}
    for w, k in csv_rows:
        can = canonical_rotation(w)
        by_len_csv.setdefault(k, set()).add(can)

    # Validate per length
    any_mismatch = False
    print(f"Validating CSV '{args.csv}' against WINNERS '{winners_path}' for lengths {Lmin}..{Lmax}")
    print()
    print(f"{'L':>2}  {'Winners':>7}  {'Found':>5}  {'Missing':>7}  {'Extras':>6}")
    print("-"*40)
    for L in range(Lmin, Lmax+1):
        win = winners_by_len.get(L, set())
        found = by_len_csv.get(L, set())
        missing = sorted(win - found)
        extras  = sorted(found - win)
        if missing or extras:
            any_mismatch = True
        print(f"{L:>2}  {len(win):>7}  {len(found):>5}  {len(missing):>7}  {len(extras):>6}")
        if missing:
            print(f"    missing: {', '.join(missing)}")
        if extras:
            print(f"    extras : {', '.join(extras)}")

    if any_mismatch and not args.ignore_mismatch:
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
