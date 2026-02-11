"""
Note: We fix the same values as in the CO2 2D example to be able to compare both
networks. This includes, e.g., fixing the layer height to be 5m which is the same as the
full height of the reservoir in the 2D example.

Note: The injection rate is fixed in the standardwell_impl.mako to calculate the
total injected gas. If changed in this file, the mako needs to be changed as well (line
2577).

Simulation parameters are loosely similar to the Utsira formation, taken from V. Singh,
A. Cavanagh, H. Hansen, B. Nazarian, M. Iding, and P. Ringrose, “Reservoir Modeling of
CO2 Plume Behavior Calibrated Against Monitoring Data From Sleipner, Norway”.

"""

from __future__ import annotations

import pathlib
from typing import Any

from pyopmnearwell.utils import units

dirname: pathlib.Path = pathlib.Path(__file__).parent

FLOW: pathlib.Path = pathlib.Path("/usr") / "bin" / "flow"
OPM_ML: pathlib.Path = pathlib.Path("/INSERT/PATH/TO/OPM_ML")
FLOW_ML: pathlib.Path = OPM_ML / "INSERT/PATH/TO/flow_gaswater_dissolution_diffuse"



# Fixed values for all runs
NUM_LAYERS: int = 5
NUM_ZCELLS: int = NUM_LAYERS * 5
# Surface density of CO2 - calculated with OPM PVT.
SURFACE_DENSITY: float = 1.86843  # unit: [kg/m^3]

##########
# Ensemble
##########
NUM_MEMBERS: int = 1


INJECTION_MIN = 1e5 * SURFACE_DENSITY     # low-end injection 
INJECTION_MAX = 8e6 * SURFACE_DENSITY     # high-end injection 

time_variables: dict[str, tuple[float, float, int]] = {
    "INJ1_DAYS": (7.0, 100.0, NUM_MEMBERS),
    "SHUT_DAYS": (7.0, 100.0, NUM_MEMBERS),
}

variables = {
    "INJECTION_RATE": (INJECTION_MIN, INJECTION_MAX, NUM_MEMBERS),
    "SCHEDULE_SEED": (0.0, 2_147_483_647, NUM_MEMBERS),
    **time_variables,
}

runspecs_ensemble: dict[str, Any] = {
    "npoints": NUM_MEMBERS,  # number of ensemble members
    "npruns": 1,             # number of parallel runs
    "variables": variables,
    "constants": {
        "PERM_0": 2e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_1": 2e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_2": 2e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_3": 2e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_4": 2e-13 * units.M2_TO_MILIDARCY,

        "INIT_PRESSURE": 80 * units.BAR_TO_PASCAL,   # <-- comma added!

        # Seabed temperature of 7°C + 35°C/km at 800–1100 m depth
        "INIT_TEMPERATURE": 40,      # [°C]
        "SURFACE_DENSITY": SURFACE_DENSITY,

        "inj": [
        [1, 1, 1, 1, 1.0],  # INJ1
        [1, 1, 1, 1, 0.0], # SHUT
        [1, 1, 1, 1, 1.0],  # INJ2
    ],
        # IMPORTANT:
        # Injection rate is now a variable -> do NOT include it in constants.

        "INJECTION_TIME": 300,  # [day]
        "REPORTSTEP_LENGTH": 1,    # [day]
        "WELL_RADIUS": 0.2,          # [m]
        "POROSITY": 0.2,
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_ZCELLS": NUM_ZCELLS,
        "NUM_XCELLS": 500,
        "LENGTH": 1000,
        "HEIGHT": 25,

        "FLOW": FLOW,
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
        "radius",
        "total_injected_volume",
        "injection_rate",
        "PI_analytical",
    ],
    "kerasify": True,
    "architecture": "fcnn",
}


#############
# Integration
#############
constants_integration_1: dict[str, Any] = {
    **runspecs_ensemble["constants"],
    **{
        "PERM_0": 7e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_1": 4e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_2": 3e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_3": 6e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_4": 2e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "INIT_PRESSURE": 65 * units.BAR_TO_PASCAL,  # unit: [Pa]
        "OPM": OPM_ML,
        "FLOW": FLOW_ML,
        # Well radius is read from the radius of the innermost grid cell of the ensemble
        # simulation (~0.23) times the ``pyopmnearwell_correction`` factor (~1.1) to
        # translate from a triangle to a radial grid. Thus it differs from the ensemble well
        # radius.
        "WELL_RADIUS": 0.25,  # unit: [m]
    },
}
# This key will be used in variables.
del constants_integration_1["NUM_ZCELLS"]

runspecs_integration_3D_and_Peaceman_1: dict[str, Any] = {
    "name": "integration_3D_and_Peaceman_1",
    "ensemble_name": "ensemble_run",
    "nn_name": "trainspecs",
    "variables": {
        "RESERVOIR_SIZE": [550] + [1100] * 6,  # unit: [m]
        "GRID_SIZE": ["20,5,5,5,5,5", 5, 10, 20, 5, 10, 20],
        "ML_MODEL_PATH": [
            "",
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

runspecs_integration_2D_1: dict[str, Any] = {
    **runspecs_integration_3D_and_Peaceman_1,
    **{
        "name": "integration_2D_1",
        "variables": {
            "RESERVOIR_SIZE": [1100] * 3,  # unit: [m]
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
    },
}

constants_integration_2: dict[str, Any] = {
    **constants_integration_1,
    **{
        "PERM_0": 8e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_1": 5e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_2": 1e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_3": 8e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_4": 5e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "INIT_PRESSURE": 90 * units.BAR_TO_PASCAL,  # unit: [Pa]
    },
}

runspecs_integration_3D_and_Peaceman_2: dict[str, Any] = {
    **runspecs_integration_3D_and_Peaceman_1,
    **{
        "name": "integration_3D_and_Peaceman_2",
        "constants": constants_integration_2,
    },
}

runspecs_integration_2D_2: dict[str, Any] = {
    **runspecs_integration_2D_1,
    **{
        "name": "integration_2D_2",
        "constants": constants_integration_2,
    },
}

constants_integration_3: dict[str, Any] = {
    **constants_integration_1,
    **{
        "PERM_0": 9e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_1": 5e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_2": 2e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_3": 5e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_4": 9e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "INIT_PRESSURE": 80 * units.BAR_TO_PASCAL,  # unit: [Pa]
    },
}

runspecs_integration_3D_and_Peaceman_3: dict[str, Any] = {
    **runspecs_integration_3D_and_Peaceman_1,
    **{
        "name": "integration_3D_and_Peaceman_3",
        "constants": constants_integration_3,
    },
}

runspecs_integration_2D_3: dict[str, Any] = {
    **runspecs_integration_2D_1,
    **{
        "name": "integration_2D_3",
        "constants": constants_integration_3,
    },
}

constants_integration_4: dict[str, Any] = {
    **constants_integration_1,
    **{
        "PERM_0": 9e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_1": 5e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_2": 2e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_3": 9e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_4": 6e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "INIT_PRESSURE": 70 * units.BAR_TO_PASCAL,  # unit: [Pa]
    },
}

runspecs_integration_3D_and_Peaceman_4: dict[str, Any] = {
    **runspecs_integration_3D_and_Peaceman_1,
    **{
        "name": "integration_3D_and_Peaceman_4",
        "constants": constants_integration_4,
    },
}

runspecs_integration_2D_4: dict[str, Any] = {
    **runspecs_integration_2D_1,
    **{
        "name": "integration_2D_4",
        "constants": constants_integration_4,
    },
}