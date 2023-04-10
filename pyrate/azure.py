from pathlib import Path
import numpy as np

def read_azure_file_sfactor(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(filepath, "r") as input:
        lines = input.readlines()
        energies = np.zeros(len(lines))
        sfactor = np.zeros(len(lines))
        for idx, line in enumerate(lines):
            entries = line.split()
            if len(entries) < 5:
                continue
            energies[idx] = float(entries[0])
            sfactor[idx] = float(entries[4])

        return (energies, sfactor)

def read_azure_file_cross(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(filepath, "r") as input:
        lines = input.readlines()
        energies = np.zeros(len(lines))
        cross = np.zeros(len(lines))
        for idx, line in enumerate(lines):
            entries = line.split()
            if len(entries) < 4:
                continue
            energies[idx] = float(entries[0])
            cross[idx] = float(entries[3])

        return (energies, cross)