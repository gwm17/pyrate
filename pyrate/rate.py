from .constants import BOLTZMANN_CONST, FINE_STRUCT_CONST, AVOGADRO, BARNTOCM2, C_CM_PER_S
from .nuclear import NucleusData
from .config import Config
from .azure import read_azure_file
from .output import write_rate_file

from math import pi
import numpy as np
from scipy.integrate import simpson

#Leading factor containing only constants in rate integrand. Compute only once
LEADING_FACTOR: float = np.sqrt(8/pi) * AVOGADRO * BARNTOCM2 * C_CM_PER_S

#Reduced mass; units in = units out
def calc_reduced_mass(m1: float, m2: float) -> float:
    return m1 * m2 / (m1 + m2)

#Boltzmann factor 1/kT; T temperature in units of GK, returns units of 1/MeV
def calc_invkT(T: float) -> float:
    return 1.0 / (BOLTZMANN_CONST * T)

#Sommerfeld parameter: 2 * pi * eta ; unitless; takes in array of energies and returns array of Sommerfelds
def calc_sommerfeld(red_mass: float, z1: int, z2: int, energies: np.ndarray) -> np.ndarray:
    return 2.0 * pi * float(z1) * float(z2) * FINE_STRUCT_CONST * np.sqrt(red_mass * 0.5 / energies)

#integrand of rate equation: energies in MeV, sfactor in MeV * barn, reduced mass in MeV, inverse kT units of 1/MeV, z1 & z2 proton numbers
#returns units of cm^3 / (mol * s)
def rate_func(energies: np.ndarray, sfactor: np.ndarray, red_mass: float, invkT: float, z1: int, z2: int) -> np.ndarray:
    const = LEADING_FACTOR * np.sqrt((invkT**3.0)/red_mass)
    return const * sfactor * np.exp(-1.0*(calc_sommerfeld(red_mass, z1, z2, energies) + energies*invkT))

#perform integral to solve for rate at a temperature T
def integrate_rate_func(T: float, projectile: NucleusData, target: NucleusData, energies: np.ndarray, sfactor: np.ndarray) -> float:
    red_mass = calc_reduced_mass(projectile.mass, target.mass)
    invkT = calc_invkT(T)
    integrand = rate_func(energies, sfactor, red_mass, invkT, projectile.Z, target.Z)
    return simpson(integrand, energies)

def calc_rate(config: Config):
    npoints = (config.T_max - config.T_min) / config.T_step
    temperatures = np.linspace(config.T_min, config.T_max, npoints)
    rates = np.zeros(len(temperatures))
    energies, sfactor = read_azure_file(config.sfactor_filepath)
    for idx, t in enumerate(temperatures):
        rates[idx] = integrate_rate_func(t, config.projectile, config.target, energies, sfactor)
    write_rate_file(config.output_filepath, temperatures, rates)