import pathlib
import random
from typing import Any

from pyopmnearwell.utils import formulas, units

dirname: pathlib.Path = pathlib.Path(__file__).parent

OPM_ML: str = "/home/peter/Documents/2023_CEMRACS/opm_ml"
FLOW_ML: str = f"{OPM_ML}/build/opm-simulators/bin/flow_gaswater_dissolution_diffuse"
OPM: str = "/home/peter/Documents/2023_CEMRACS/opm"
FLOW: str = f"{OPM}/build/opm-simulators/bin/flow"


# Fixed values for all runs
num_layers: int = 5
reservoir_height: float = 20.0  # unit: [m]
well_radius: float = 0.8  # unit: [m]
porosity: float = 0.36  # unit: [-]

injection_rate: float = 5e6  # unit: [m^3/d]
injection_time: float = 5.0  # unit: [d]
surface_density: float = 1.86843  # unit: [kg/m^3]
init_temperature: float = 25  # unit: [°C]

##########
# Ensemble
##########
npoints: int = 500
num_zcells: int = num_layers * 5
num_xcells: int = 40

# NOTE: Vertical permeabilties are equal to horizontal permeability.
variables: dict[str, tuple[float, float, int]] = {
    f"PERM_{i}": (
        5e-13 * units.M2_TO_MILIDARCY,
        5e-12 * units.M2_TO_MILIDARCY,
        npoints,
    )  # unit: [mD]
    for i in range(num_layers)
}


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
    "INIT_TEMPERATURE": init_temperature,  # unit: [°C])
    "SURFACE_DENSITY": surface_density,  # unit: [kg/m^3]
    "INJECTION_RATE": injection_rate * surface_density,  # unit: [kg/d]
    # NOTE: 1e2 m^3/day is approx 2L/s for each meter of well.
    "INJECTION_TIME": injection_time,  # unit: [day]
    "WELL_RADIUS": well_radius,  # unit: [m]; Fixed during training.
    "POROSITY": porosity,  # unit [-]
    #
    "NUM_LAYERS": num_layers,
    "NUM_ZCELLS": num_zcells,
    "NUM_XCELLS": num_xcells,
    "LENGTH": 100,
    "HEIGHT": reservoir_height,
    "FLOW": FLOW,
    "OPM": OPM,
}

runspecs_ensemble: dict[str, Any] = {
    "name": "ensemble_1",
    "npoints": npoints,  # number of ensemble members
    "npruns": min(npoints, 5),  # number of parallel runs
    "variables": variables,
    "constants": constants,
}

##########
# Training
##########
trainspecs: dict[str, Any] = {
    "name": "trainspecs",
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": False,  # True, False
    "WI_log": False,  # True, False
    "pressure_padding": "neighbor",  # zeros, init, neighbor
    "saturation_padding": "zeros",  # zeros, init, neighbor
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
        "radius",
        "total_injected_volume",
        "PI_analytical",
    ],
    #
    "kerasify": True,
}


##########
# Integration
##########
constants: dict[str, Any] = {
    f"PERM_{i}": random.uniform(5e-13, 5e-12) * units.M2_TO_MILIDARCY
    for i in range(num_layers)
}
constants.update
constants.update(
    {
        "INIT_PRESSURE": 65 * units.BAR_TO_PASCAL,  # unit: [Pa]
        "INIT_TEMPERATURE": init_temperature,  # unit: [°C]
        #
        "INJECTION_RATE": injection_rate * surface_density,  # unit: [kg/d]
        "INJECTION_TIME": injection_time,  # unit: [day]
        "WELL_RADIUS": well_radius,  # unit: [m]; Fixed during training.
        "POROSITY": porosity,  # unit [-]
        #
        "RESERVOIR_SIZE": 5100,  # unit: [m]
        "HEIGHT": reservoir_height,  # unit: [m]
        "NUM_LAYERS": num_layers,
        # In comparison to the fine scale example the number of zcells is reduced.
        "NUM_ZCELLS": num_layers,
        "OPM_ML": OPM_ML,
        "FLOW_ML": FLOW_ML,
    }
)
runspecs_integration: dict[str, Any] = {
    "variables": {
        # "GRID_SIZE": [25, 100, 200, 25, 100, 200],
        # "RUN_NAME": [
        #     "100x100m_NN",
        #     "25x25m_NN",
        #     "12.5x12.5m_NN",
        #     "100x100m_Peaceman",
        #     "25x25m_Peaceman",
        #     "12.5x12.5m_Peaceman",
        # ],
        "GRID_SIZE": [25],
        "RUN_NAME": ["100x100m_NN"],
    },
    "constants": constants,
}
