import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

#Simple function to read azure energy, s-factor data. Energies in MeV, s-factor in MeV*barn
def read_azure_file(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
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

energy, sfactor = read_azure_file("my_azure_data.txt")
fig, ax = plt.subplots(1,1)
ax.plot(energy, sfactor, label="data")
ax.set_ylabel("S-Factor")
ax.set_xlabel("Energy")
fig.legend()
plt.show()