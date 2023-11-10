import copy
import random
from typing import Any

from pyopmnearwell.utils import formulas, units

OPM_ML: str = "/home/peter/Documents/2023_CEMRACS/opm_ml"
FLOW_ML: str = f"{OPM_ML}/build/opm-simulators/bin/flow_gaswater_dissolution_diffuse"
OPM: str = "/home/peter/Documents/2023_CEMRACS/opm"
FLOW: str = f"{OPM}/build/opm-simulators/bin/flow"

npoints: int = 50

num_layers: int = 5
num_zcells: int = num_layers * 5
injection_rate: float = 0.04 * units.Q_per_seconds_to_Q_per_day  # unit: [m^3/d]
injection_rate: float = 5e6  # unit: [m^3/d]
surface_density: float = 1.86843  # unit: [kg/m^3]


# NOTE: Vertical permeabilties are equal to horizontal permeability.
# variables_1: small differences in permeability.
variables_1: dict[str, tuple[float, float, int]] = {
    f"PERM_{i}": (  # unit: [mD]
        5e-13 * units.M2_TO_MILIDARCY,
        5e-12 * units.M2_TO_MILIDARCY,
        npoints,
    )
    for i in range(num_layers)
}

for variables in [variables_1]:
    variables.update(
        {
            "INIT_PRESSURE": (
                50 * units.BAR_TO_PASCAL,
                150 * units.BAR_TO_PASCAL,
                npoints,
            ),  # unit: [Pa]
        }
    )

constants: dict[str, Any] = {
    "INIT_TEMPERATURE": 25,  # unit: [°C])
    "SURFACE_DENSITY": surface_density,  # unit: [kg/m^3]
    "INJECTION_RATE": injection_rate * surface_density,  # unit: [kg/d]
    # NOTE: 1e2 m^3/day is approx 2L/s for each meter of well.
    "INJECTION_TIME": 5,  # unit: [day]
    "WELL_RADIUS": 0.25,  # unit: [m]; Fixed during training.
    "POROSITY": 0.36,  # unit [-]
    #
    "NUM_LAYERS": num_layers,
    "NUM_ZCELLS": num_zcells,
    "LENGTH": 100,
    "HEIGHT": 20,
    "FLOW": FLOW,
    "OPM": OPM,
}


runspecs_ensemble_1: dict[str, Any] = {
    "name": "ensemble_1",
    "npoints": npoints,  # number of ensemble members
    "npruns": min(npoints, 5),  # number of parallel runs
    "variables": variables_1,
    "constants": constants,
}

trainspecs_1: dict[str, Any] = {
    "name": "trainspecs_1",
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": True,  # True, False
    "WI_log": True,  # True, False
    "pressure_padding": "neighbor",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros, epsilon
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "features": [
        "pressure_upper",
        "pressure",
        "pressure_lower",
        "saturation_upper",
        "saturation",
        "saturation_lower",
        "permeability_upper",
        "permeability",
        "permeability_lower",
        "total_injected_volume",
        "WI_analytical",
    ],
    #
    "kerasify": True,
}

trainspecs_2: dict[str, Any] = {
    "name": "trainspecs_2",
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": False,  # True, False
    "WI_log": False,  # True, False
    "pressure_padding": "neighbor",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros, epsilon
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    # Network architecture
    "features": [
        "pressure_upper",
        "pressure",
        "pressure_lower",
        "saturation_upper",
        "saturation",
        "saturation_lower",
        "permeability_upper",
        "permeability",
        "permeability_lower",
        "total_injected_volume",
        "WI_analytical",
    ],
    #
    "kerasify": True,
}
