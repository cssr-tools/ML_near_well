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

OPM: pathlib.Path = pathlib.Path("/opt") / "opm_src"
FLOW: pathlib.Path = pathlib.Path("/usr") / "local" / "bin" / "flow"


# Fixed values for all runs
NUM_LAYERS: int = 5
NUM_ZCELLS: int = NUM_LAYERS * 5
# Surface density of CO2 - calculated with OPM PVT.
SURFACE_DENSITY: float = 1.86843  # unit: [kg/m^3]

##########
# Ensemble
##########
NUM_MEMBERS: int = 20

INJECTION_MIN = 1e5 * SURFACE_DENSITY  # low-end injection ~190 t/d
INJECTION_MAX = 6e6 * SURFACE_DENSITY  # high-end injection ~11 200 t/d
# INJECTION_MIN = 7.5e5 * SURFACE_DENSITY   # ~1400 t/d
# INJECTION_MAX = 2.2e6 * SURFACE_DENSITY   # ~4100 t/d

time_variables: dict[str, tuple[float, float, int]] = {
    "INJ1_DAYS": (5.0, 100.0, NUM_MEMBERS),
    "SHUT_DAYS": (7.0, 42.0, NUM_MEMBERS),
}
"""
time_variables: dict[str, tuple[float, float, int]] = {
    "INJ1_DAYS": (5.0, 30.0, NUM_MEMBERS), 
    "SHUT_DAYS": (7.0, 15.0, NUM_MEMBERS),
}
"""
variables = {
    "INJECTION_RATE": (INJECTION_MIN, INJECTION_MAX, NUM_MEMBERS),
    "SCHEDULE_SEED": (0.0, 2_147_483_647, NUM_MEMBERS),
    **time_variables,
}
runspecs_ensemble: dict[str, Any] = {
    "npoints": NUM_MEMBERS,  # number of ensemble members
    "npruns": 5,  # number of parallel runs
    "variables": variables,
    "constants": {
        "PERM_0": 4e-13 * units.M2_TO_MILIDARCY,
        "PERM_1": 8e-13 * units.M2_TO_MILIDARCY,
        "PERM_2": 1.2e-12 * units.M2_TO_MILIDARCY,
        "PERM_3": 1.6e-12 * units.M2_TO_MILIDARCY,
        "PERM_4": 2e-12 * units.M2_TO_MILIDARCY,
        "INIT_PRESSURE": 80 * units.BAR_TO_PASCAL,
        "INIT_TEMPERATURE": 40,  # [°C]
        "SURFACE_DENSITY": SURFACE_DENSITY,
        "inj": [
            [1, 1, 1, 1, 1.0],  # INJ1
            [1, 1, 1, 1, 0.0],  # SHUT
            [1, 1, 1, 1, 1.0],  # INJ2
        ],
        "INJECTION_TIME": 180,  # [day]
        "HISTORY_WINDOW_DAYS": 180,  # [day] s
        "REPORTSTEP_LENGTH": 0.5,  # [day]
        "WELL_RADIUS": 0.2,  # [m]
        "POROSITY": 0.2,
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_ZCELLS": NUM_ZCELLS,
        "NUM_XCELLS": 200,
        "LENGTH": 400,
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
    # Network architecture.
    # NOTE feature-1 is the upper neighbor and feature+1 is the lower neighbor.
    "features": [
        "pressure-1",
        "pressure+0",
        "pressure+1",
        "saturation-1",
        "saturation+0",
        "saturation+1",
        "equivalent_radius",
        "tot_inj_gas",
        "injection_rate",
        "current_injection_time",
        "previous_shutin_time",
        "previous_injection_time",
        "older_history_time",
        "analytical_PI",
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
        "OPM": OPM,
        "FLOW": FLOW,
        "WELL_RADIUS": 0.2,
        "INJ1_DAYS": 75.0,
        "SHUT_DAYS": 33.0,
        "INJECTION_RATE": 9.0e6 * SURFACE_DENSITY,
    },
}
# This key will be used in variables.
del constants_integration_1["NUM_ZCELLS"]
# runspecs_integration_3D_and_Peaceman_1: dict[str, Any] = {
#     "name": "integration_3D_and_Peaceman_1",
#     "ensemble_name": "ensemble_run",
#     "nn_name": "trainspecs",
#     "variables": {
#         "RESERVOIR_SIZE": [550] + [1100] * 6,  # unit: [m]
#         "GRID_SIZE": ["20,5,5,5,5,5", 5, 10, 20, 5, 10, 20],
#         "MLNEARWELLCONFIGFILE": [
#             "",
#             str(dirname / "nn" / "MLNearWellConfig.json"),
#             str(dirname / "nn" / "MLNearWellConfig.json"),
#             str(dirname / "nn" / "MLNearWellConfig.json"),
#             "",
#             "",
#             "",
#         ],
#         "RUN_NAME": [
#             "8x8M_Peaceman_more_zcells",
#             "90x90m_NN_3D",
#             "52x52m_NN_3D",
#             "27x27m_NN_3D",
#             "90x90m_Peaceman",
#             "52x52m_Peaceman",
#             "27x27m_Peaceman",
#         ],
#         "USEMLNEARWELL": [
#             "",
#             "--UseMLNearWell=true",
#             "--UseMLNearWell=true",
#             "--UseMLNearWell=true",
#             "",
#             "",
#             "",
#         ],
#         "NUM_ZCELLS": [NUM_LAYERS * 5] + [NUM_LAYERS] * 6,
#     },
#     "constants": constants_integration_1,
# }

runspecs_integration_3D_and_Peaceman_1: dict[str, Any] = {
    "name": "integration_3D_and_Peaceman_1",
    "ensemble_name": "ensemble_run",
    "nn_name": "trainspecs",
    "variables": {
        # "RESERVOIR_SIZE": [550, 1100, 1100, 1100, 1100, 1100, 1100],
        # # Benchmark får LGR/telescopic grid.
        # # De andre er uniform coarse grid.
        # "GRID_SIZE": ["20,5,5,5,5,5", 5, 10, 20, 5, 10, 20],
        "RESERVOIR_SIZE": [1100],  # 1100, 1100, 1100, 1100, 1100],
        # Benchmark får LGR/telescopic grid.
        # De andre er uniform coarse grid.
        "GRID_SIZE": [5],  # 10, 20, 5, 10, 20],
        # "GRID_XFACTOR": [1.53, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "MLNEARWELLCONFIGFILE": [
            # "",
            str(dirname / "nn" / "MLNearWellConfig.json"),
            # str(dirname / "nn" / "MLNearWellConfig.json"),
            # str(dirname / "nn" / "MLNearWellConfig.json"),
            # "",
            # "",
            # "",
        ],
        "RUN_NAME": [
            # "8x8M_Peaceman_more_zcells",
            "90x90m_NN",
            # "52x52m_NN",
            # "27x27m_NN",
            # "90x90m_Peaceman",
            # "52x52m_Peaceman",
            # "27x27m_Peaceman",
        ],
        "USEMLNEARWELL": [
            # "",
            "--UseMLNearWell=true",
            # "--UseMLNearWell=true",
            # "--UseMLNearWell=true",
            # "",
            # "",
            # "",
        ],
        # "NUM_ZCELLS": [NUM_LAYERS * 5] + [NUM_LAYERS] * 6,
        "NUM_ZCELLS": [NUM_LAYERS],
    },
    "constants": constants_integration_1,
}

constants_integration_2: dict[str, Any] = {
    **constants_integration_1,
    **{
        "PERM_0": 5e-13 * units.M2_TO_MILIDARCY,
        "PERM_1": 1e-12 * units.M2_TO_MILIDARCY,
        "PERM_2": 1.5e-12 * units.M2_TO_MILIDARCY,
        "PERM_3": 2e-12 * units.M2_TO_MILIDARCY,
        "PERM_4": 3e-12 * units.M2_TO_MILIDARCY,
        "INIT_PRESSURE": 80 * units.BAR_TO_PASCAL,  # unit: [Pa]
    },
}

runspecs_integration_3D_and_Peaceman_2: dict[str, Any] = {
    **runspecs_integration_3D_and_Peaceman_1,
    **{
        "name": "integration_3D_and_Peaceman_2",
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
