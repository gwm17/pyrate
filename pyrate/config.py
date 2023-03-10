from dataclasses import dataclass, field
from .nuclear import NucleusData
from pathlib import Path
from typing import Optional
import json

from .nuclear import NuclearDataMap
@dataclass
class Config:
    projectile: NucleusData = field(default_factory=NucleusData)
    target: NucleusData = field(default_factory=NucleusData)
    sfactor_filepath: Path = field(default_factory=Path)
    output_filepath: Path = field(default_factory=Path)
    T_min: float = 0.0
    T_max: float = 0.0
    T_step: float = 0.0

def read_config_file(file: str) -> Optional[Config]:
    filepath = Path(file)
    if not filepath.exists():
        print(f"Could not read configuration from file {filepath}! File does not exist")
        return None
    
    nuclear_data = NuclearDataMap()

    with open(filepath, "r") as input:
        config = Config()
        data = json.load(input)
        zp = data["Z_projectile"]
        ap = data["A_projectile"]
        zt = data["Z_target"]
        at = data["A_target"]
        config.projectile = nuclear_data.get_data(zp, ap)
        config.target = nuclear_data.get_data(zt, at)
        config.sfactor_filepath = Path(data["Sfactor_file"])
        if not config.sfactor_filepath.exists():
            print(f"Azure data file {config.sfactor_filepath} does not exist!")
            return None
        config.output_filepath = Path(data["output_file"])
        config.T_min = data["T_min"]
        config.T_max = data["T_max"]
        config.T_step = data["T_step"]
        return config