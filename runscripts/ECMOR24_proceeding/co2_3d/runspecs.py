"""

Note: We fix the same values as in the CO2 2D example to be able to compare both
networks. This includes, e.g., fixing the layer height to be 5m which is the same as the
full height of the reservoir in the 2D example.

Note: The injection time is shorter than in the 2D example to avoid the CO2 overspilling
into the pore volume cell at the boundary of the near-well simulation.


Simulation parameters are loosely similar to the Utsira formation, taken from V. Singh,
A. Cavanagh, H. Hansen, B. Nazarian, M. Iding, and P. Ringrose, “Reservoir Modeling of
CO2 Plume Behavior Calibrated Against Monitoring Data From Sleipner, Norway”.

"""

import pathlib
from typing import Any

from pyopmnearwell.utils import units

dirname: pathlib.Path = pathlib.Path(__file__).parent

OPM_ML: str = "/home/peter/Documents/2023_CEMRACS/opm_ml"
FLOW_ML: str = f"{OPM_ML}/build/opm-simulators/bin/flow_gaswater_dissolution_diffuse"
OPM: str = "/home/peter/Documents/2023_CEMRACS/opm"
FLOW: str = f"{OPM}/build/opm-simulators/bin/flow"


# Fixed values for all runs
NUM_LAYERS: int = 5
# Utsira formation has a height of ~300 m and connection length of 38 m (horizontal?).
# We consider only the area along the connection.
RESERVOIR_HEIGHT: float = 25.0  # unit: [m]
# The well is fully inside the innermost cell, which we consider as the "well"
# to ensure easier computation. The "well" radius for computation is thus
# dependent on the cell size and this value gets disregarded when creating the
# dataset.
WELL_RADIUS: float = 0.25  # unit: [m]
POROSITY: float = 0.35  # unit: [-]

# Surface rates for Utsira are 3e4 - 6e5 m^3/d on a perforation length of 38 m.
# We take larger values on a perforation length of 25 m to obtain a slightly
# larger pressure gradient.
# NOTE: The injection rate is 5x the injection rate for the CO2 2D example to be able to
# compare both models.
# NOTE: 1e2 m^3/day is approx 2L/s for each meter of well.
INJECTION_RATE: float = 5e6  # unit: [m^3/d]
INJECTION_TIME: float = 10.0  # unit: [d]
# Surface density of CO2 - calculated with OPM PVT.
SURFACE_DENSITY: float = 1.86843  # unit: [kg/m^3]
# Seabed temperature of 7°C + 35°C/km at 800-1000m depth ~ 40°C
INIT_TEMPERATURE: float = 25  # unit: [°C]

##########
# Ensemble
##########
NUM_MEMBERS: int = 200
NUM_ZCELLS: int = NUM_LAYERS * 5
NUM_XCELLS: int = 50

# Permeability ranges for each layer.
variables: dict[str, tuple[float, float, int]] = {
    # Permeability of Utsira formation is 1000 - 5000 mD ~ 1e-12 - 5e-12 m^2.
    # Anisotropy ratio is 0.1. The vertical permeability is calculated from the
    # horizontal permeability in the ensemble.mako and integration.mako file.s
    f"PERM_{i}": (
        5e-13 * units.M2_TO_MILIDARCY,
        1e-11 * units.M2_TO_MILIDARCY,
        NUM_MEMBERS,
    )  # unit: [mD]
    for i in range(NUM_LAYERS)
}

variables.update(
    {
        # Assumed pressure regime of Utsira formation.
        "INIT_PRESSURE": (
            50 * units.BAR_TO_PASCAL,
            120 * units.BAR_TO_PASCAL,
            NUM_MEMBERS,
        ),  # unit: [Pa]
    }
)

runspecs_ensemble: dict[str, Any] = {
    "npoints": NUM_MEMBERS,  # number of ensemble members
    "npruns": 5,  # number of parallel runs
    "variables": variables,
    "constants": {
        "INIT_TEMPERATURE": INIT_TEMPERATURE,  # unit: [°C])
        "SURFACE_DENSITY": SURFACE_DENSITY,  # unit: [kg/m^3]
        "INJECTION_RATE": INJECTION_RATE * SURFACE_DENSITY,  # unit: [kg/d]
        "INJECTION_TIME": INJECTION_TIME,  # unit: [day]
        "REPORTSTEP_LENGTH": 0.1,  # unit [d]
        "WELL_RADIUS": WELL_RADIUS,  # unit: [m]; Fixed during training.
        "POROSITY": POROSITY,  # unit [-]
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_ZCELLS": NUM_ZCELLS,
        "NUM_XCELLS": NUM_XCELLS,
        "LENGTH": 100,
        "HEIGHT": RESERVOIR_HEIGHT,
        "FLOW": FLOW,
        "OPM": OPM,
    },
}


##########
# Training
##########
trainspecs: dict[str, Any] = {
    "name": "trainspecs",
    # Data conversion/padding
    "pressure_unit": "Pascal",  # bar, Pascal
    "permeability_log": False,  # True, False
    "WI_log": True,  # True, False
    "pressure_padding": "neighbor",  # zeros, init, neighbor
    "saturation_padding": "zeros",  # zeros, init, neighbor
    "permeability_padding": "zeros",  # zeros, epsilon
    # Scaling/normalization
    "MinMax_scaling": True,
    "Z-normalization": False,
    "percentage_loss": False,
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
    "kerasify": True,
    "architecture": "fcnn",
}


##########
# Integration
##########
constants_integration_1: dict[str, Any] = runspecs_ensemble["constants"] | {
    "PERM_0": 7e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "PERM_1": 4e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "PERM_2": 3e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "PERM_3": 6e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "PERM_4": 2e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "INIT_PRESSURE": 65 * units.BAR_TO_PASCAL,  # unit: [Pa]
    "RESERVOIR_SIZE": 1100,  # unit: [m]
    "OPM": OPM_ML,
    "FLOW": FLOW_ML,
}
# This key will be used in variables.
del constants_integration_1["NUM_ZCELLS"]

runspecs_integration_3D_and_Peaceman_1: dict[str, Any] = {
    "name": "integration_3D_and_Peaceman_1",
    "ensemble_name": "ensemble",
    "nn_name": "trainspecs",
    "variables": {
        "GRID_SIZE": [90, 6, 10, 20, 6, 10, 20],
        "ML_MODEL_PATH": [
            "",
            str(dirname / "nn" / "WI.model"),
            str(dirname / "nn" / "WI.model"),
            str(dirname / "nn" / "WI.model"),
            str(dirname / "nn" / "WI.model"),
            "",
            "",
            "",
        ],
        "RUN_NAME": [
            "8x8M_Peaceman_more_zcells",
            "90x90m_NN_3D",
            "52x52m_NN_3D",
            "27x27m_NN_3D",
            "90x90m_Peaceman",
            "52x52m_Peaceman",
            "27x27m_Peaceman",
        ],
        "NUM_ZCELLS": [NUM_LAYERS * 5] + [NUM_LAYERS] * 6,
    },
    "constants": constants_integration_1,
}

runspecs_integration_2D_1: dict[str, Any] = runspecs_integration_3D_and_Peaceman_1 | {
    "name": "integration_2D_1",
    "variables": {
        "GRID_SIZE": [6, 10, 20],  # 55],  # , 55],
        "ML_MODEL_PATH": [
            str(dirname / ".." / "co2_2d" / "nn" / "WI.model"),
            str(dirname / ".." / "co2_2d" / "nn" / "WI.model"),
            str(dirname / ".." / "co2_2d" / "nn" / "WI.model"),
        ],
        "RUN_NAME": [
            "90x90m_NN_2D",
            "52x52m_NN_2D",
            "27x27m_NN_2D",
        ],
        "NUM_ZCELLS": [NUM_LAYERS] * 3,
    },
}

constants_integration_2: dict[str, Any] = constants_integration_1 | {
    "PERM_0": 9e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "PERM_1": 5e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "PERM_2": 2e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "PERM_3": 9e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "PERM_4": 6e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "INIT_PRESSURE": 70 * units.BAR_TO_PASCAL,  # unit: [Pa]
}

runspecs_integration_3D_and_Peaceman_2: dict[str, Any] = (
    runspecs_integration_3D_and_Peaceman_1
    | {
        "name": "integration_3D_and_Peaceman_2",
        "constants": constants_integration_2,
    }
)

runspecs_integration_2D_2: dict[str, Any] = runspecs_integration_2D_1 | {
    "name": "integration_2D_2",
    "constants": constants_integration_2,
}

constants_integration_3: dict[str, Any] = constants_integration_1 | {
    "PERM_0": 9e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "PERM_1": 5e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "PERM_2": 2e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "PERM_3": 5e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "PERM_4": 9e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
    "INIT_PRESSURE": 80 * units.BAR_TO_PASCAL,  # unit: [Pa]
}

runspecs_integration_3D_and_Peaceman_3: dict[str, Any] = (
    runspecs_integration_3D_and_Peaceman_1
    | {
        "name": "integration_3D_and_Peaceman_3",
        "constants": constants_integration_3,
    }
)

runspecs_integration_2D_3: dict[str, Any] = runspecs_integration_2D_1 | {
    "name": "integration_2D_3",
    "constants": constants_integration_3,
}
