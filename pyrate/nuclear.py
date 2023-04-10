import numpy as np
from dataclasses import dataclass

PATH_TO_MASSFILE = "./etc/amdc2016_mass.txt"

@dataclass
class NucleusData:
    mass: float = 0.0 #MeV
    mass_u: float = 0.0 #u
    elementSymbol: str = ""
    isotopicSymbol: str = ""
    prettyIsotopicSymbol: str = ""
    Z: int = 0
    A: int = 0

    def __str__(self):
        return self.isotopicSymbol

    def get_latex_rep(self):
        return "$^{" + str(self.A) + "}$" + self.elementSymbol

def generate_nucleus_id(z: np.uint32, a: np.uint32) -> np.uint32 :
    return z*z + z + a if z > a else a*a + z

class NuclearDataMap:
    U2MEV: float = 931.49410242
    ELECTRON_MASS: float = 0.000548579909

    def __init__(self):
        self.map = {}

        with open(PATH_TO_MASSFILE) as massfile:
            massfile.readline()
            massfile.readline()
            for line in massfile:
                entries = line.split()
                data = NucleusData()
                data.Z = int(entries[1])
                data.A = int(entries[2])
                data.mass_u = (float(entries[4])  + 1.0e-6 * float(entries[5]) - float(data.Z) * self.ELECTRON_MASS)
                data.mass = data.mass_u * self.U2MEV
                data.elementSymbol = entries[3]
                data.isotopicSymbol = f"{data.A}{entries[3]}"
                data.prettyIsotopicSymbol = f"<sup>{data.A}</sup>{entries[3]}"
                self.map[generate_nucleus_id(data.Z, data.A)] = data

    def get_data(self, z: np.uint32, a: np.uint32) -> NucleusData:
        return self.map[generate_nucleus_id(z, a)]
