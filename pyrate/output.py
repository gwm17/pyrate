from pathlib import Path
import numpy as np

def write_rate_file(filepath: Path, temperature: np.ndarray, rate: np.ndarray):
    with open(filepath, "w") as output:
        output.write("Temperature(GK),Rate(cm^3/(mol*s))\n")
        lines = [f"{t:.3f},{rate[idx]:.6e}\n" for idx, t in temperature]
        output.writelines(lines)