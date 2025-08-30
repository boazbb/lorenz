from dataclasses import dataclass
import numpy as np
from typing import Dict
import json

@dataclass
class PeriodicOrbit:
    symbol_sequence: str
    length: int
    stable_characteristic_multiplier: float

def main():
    with open("periodic_orbits_by_length.json", "r") as f:
        data = json.load(f)
    
    dm_guesses = np.linspace(1.0, 1.15, 16)

    for m in [4, 5, 6, 7, 8]:
        print('=' * 10)
        print(f"m={m}")
        for dm_guess in dm_guesses:
            sum = calc_sum(data, m, dm_guess)
            print(f"dm_guess={dm_guess}, sum={sum}")

def calc_sum(data: Dict[int, list[PeriodicOrbit]], m: int, dm_guess: float):
    sum = 0.0
    divisors = get_all_divisors(m)
    for divisor in divisors:
        for orbit in data[str(divisor)]:
            # print(f"Adding {orbit['length'] * 2} points")
            # print(f"SCM: {orbit['stable_characteristic_multiplier']}")
            scm = orbit["stable_characteristic_multiplier"]
            num_points = orbit["length"]
            sum += num_points * (1 / scm) * (scm ** dm_guess)
    return sum

def get_all_divisors(m: int) -> list[int]:
    return [i for i in range(2, m+1) if m % i == 0]


if __name__ == "__main__":
    main()