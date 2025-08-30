# lorenz

Python scripts for computing periodic orbit in the lorenz system, and computing the hausdorff dimension.

Draw inspiration from [The fractal property of the Lorenz attractor](https://www.sciencedirect.com/science/article/abs/pii/S0167278903004093).

## Data
The data from the original papers can be download from [here](https://dept.math.lsa.umich.edu/~divakar/lorenz/data.zip).

The scripts can check our output against it, visualize those orbits and compute the monodromies, multipliers and hausdorff dimension from it. The scripts expect it to be in the root dir of this project.

layout:

```
data/
  DATA1/
    orbit{N}.dat   # 3 space-separated lines: x, y, z samples (one closed orbit)
    T{N}.dat       # single float: the period of orbit N
  DATA2/
    WINNERS        # lines: "orbit <N>: <A/B symbol sequence>"
```

> **Note:** `find_orbits.py` does **not** require the data bundle; it discovers orbits directly from the ODE and writes a CSV with the ones it finds.

## Environment

- Python 3.10+
- `pip install numpy scipy matplotlib tqdm pandas`

---

## Scripts

### `find_orbits.py` — enumerate periodic orbits by A/B words (Option A, return map)

Runs the process for a long time, then enumerates all words, uses their occurence in the run as seeds and try to optimize a periodic orbit from them.

**Examples**

```bash
# collect 5k section hits and solve words of length 2..4 (this one actually return all the periodic orbits)
python find_orbits.py --min_len 2 --max_len 4 --hits 5000 --seeds 100 --verbose
```

Outputs a pretty table and a CSV: `lorenz_orbits_len_<min>_<max>.csv` (columns: word, k, canonical_word, period, x_at_z27, y_at_z27, labels). The arguments "--save_hits" and "--load_hits" can save some time when running the scripts again.

---

### `validate_orbits.py` — check your CSV against the reference “WINNERS” list

Parses the `WINNERS` file (`data/DATA2/WINNERS`) and your CSV, then compares **by length** using **canonical rotation** (e.g., `AB ≡ BA`). Reports counts per length and lists any **missing** winners and **extras** not in the reference.

**Examples**
```
python validate_orbits.py lorenz_orbits_len_2_4.csv
```
---

### `viz_orbit.py` — plot an orbit, the plane z=27, and the section hits

Loads `data/DATA1/orbit{N}.dat`, plots the 3D orbit and the plane `z=27`, and overlays the intersection points (optionally one direction). It also reads the symbol sequence for orbit `N` from `data/DATA2/WINNERS` and puts it in the title/console.

**Example**

```bash
# Visualize orbit #123 from the dataset
python viz_orbit.py 123
```

---

### `preprocess_orbits.py` — compute stability and bundle by word length

Iterates over the dataset orbits listed in `WINNERS`, integrates the Lorenz **variational equations** over one period to get the monodromy matrix, extracts the **stable characteristic multiplier** (|λ_s|), and writes a JSON mapping `length → [PeriodicOrbit]` to `periodic_orbits_by_length.json`.

**Example**

```bash
python preprocess_orbits.py
# → periodic_orbits_by_length.json
```

Each saved record contains: `{symbol_sequence, length, stable_characteristic_multiplier}`.

---

### `compute_hausdorff.py` — sweep a dimension guess and accumulate the sum

Loads `periodic_orbits_by_length.json` and for each `m` (word length) computes the sum used in the dimension estimate, iterating over divisors of `m` and the stable multipliers of the corresponding orbits. Prints the sum for a grid of dimension guesses (e.g., 1.00 … 1.15).