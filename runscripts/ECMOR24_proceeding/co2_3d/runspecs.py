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
NUM_MEMBERS: int = 200

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
        # Seabed temperature of 7°C + 35°C/km at 800-1100m depth ~ 40°C
        "INIT_TEMPERATURE": 40,  # unit: [°C]
        "SURFACE_DENSITY": SURFACE_DENSITY,  # unit: [kg/m^3]
        # Surface rates for Utsira are 3e4 - 6e5 m^3/d on a perforation length of 38 m.
        # We take larger values on a perforation length of 25 m to obtain a slightly
        # larger pressure gradient.
        # NOTE: The injection rate is 5x the injection rate for the CO2 2D example
        # (which has a height of 5 m) to be able to compare both models.
        # NOTE: 1e2 m^3/day is approx 2L/s for each meter of well.
        "INJECTION_RATE": 5e6 * SURFACE_DENSITY,  # unit: [kg/d]
        "INJECTION_TIME": 10.0,  # unit: [day]
        "REPORTSTEP_LENGTH": 0.1,  # unit [d]
        # The well is fully inside the innermost cells, which we consider as the "well"
        # for easier computation of the data-driven well index. Therefore the well index
        # for fine-scale simulation and integration differ.
        "WELL_RADIUS": 0.2,  # unit: [m]
        # Porosity ot Utsira formation is 0.34 - 0.36
        "POROSITY": 0.35,  # unit: [-]
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_ZCELLS": NUM_ZCELLS,
        "NUM_XCELLS": 50,
        "LENGTH": 100,
        # Utsira formation has a height of ~300 m and connection length of 38 m
        # (horizontal?).  We consider only the area along the connection.
        "HEIGHT": 25,  # unit: [m]
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
        "permeability-1",
        "permeability+0",
        "permeability+1",
        "radius",
        "tot_inj_gas",
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
        "PERM_0": 7e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_1": 4e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_2": 3e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_3": 6e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERM_4": 2e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "INIT_PRESSURE": 65 * units.BAR_TO_PASCAL,  # unit: [Pa]
        "OPM": OPM,
        "FLOW": FLOW,
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
    "ensemble_name": "ensemble",
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
