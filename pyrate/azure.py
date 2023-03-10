from pathlib import Path
import numpy as np

def read_azure_file(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(filepath, "r") as input:
        lines = input.readlines()
        energies = np.zeros(len(lines))
        sfactor = np.zeros(len(lines))
        for idx, line in enumerate(lines):
            entries = line.split()
            energies[idx] = float(entries[0])
            sfactor[idx] = float(entries[4])

        return (energies, sfactor)