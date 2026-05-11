from __future__ import annotations

import pathlib
from typing import Any

from pyopmnearwell.utils import units

dirname: pathlib.Path = pathlib.Path(__file__).parent

OPM: pathlib.Path = pathlib.Path("/opt") / "opm_src"
FLOW: pathlib.Path = pathlib.Path("/usr") / "local" / "bin" / "flow"

NUM_MEMBERS: int = 2
SURFACE_DENSITY: float = 998.414

runspecs_ensemble: dict[str, Any] = {
    "npoints": NUM_MEMBERS,  # number of ensemble members
    "npruns": 2,  # number of parallel runs
    "variables": {
        "INIT_PRESSURE": (
            50 * units.BAR_TO_PASCAL,
            150 * units.BAR_TO_PASCAL,
            NUM_MEMBERS,
        ),  # unit: [Pa]
        "INT_HEIGHT": (2, 20, NUM_MEMBERS),  # unit: [m]
        "PERM": (
            1e-14 * units.M2_TO_MILIDARCY,
            1e-12 * units.M2_TO_MILIDARCY,
            NUM_MEMBERS,
        ),  # unit: [mD]
    },
    "constants": {
        "INIT_TEMPERATURE": 40,  # unit: [°C]
        "POROSITY": 0.35,  # unit: [-]
        "SURFACE_DENSITY": SURFACE_DENSITY,  # unit: [kg/m^3]
        "INJECTION_RATE": 6e2 * SURFACE_DENSITY,  # unit: [kg/d]
        "INJECTION_TIME": 10,  # unit: [d]
        "REPORTSTEP_LENGTH": 0.1,  # unit [d]
        "NUM_XCELLS": 50,
        "LENGTH": 100,
        # The well is fully inside the innermost cells, which we consider as the "well"
        # for easier computation of the data-driven well index. Therefore the well index
        # for fine-scale simulation and integration differ.
        "WELL_RADIUS": 0.2,  # unit: [m]
        "FLOW": FLOW,
    },
}

##########
# Training
##########
trainspecs: dict[str, Any] = {
    "features": ["pressure", "permeability", "height", "equivalent_radius"],
    "MinMax_scaling": True,
    "kerasify": True,
    "architecture": "fcnn",
    "permeability_log": False,
}

#############
# Integration
#############
runspecs_integration_1: dict[str, Any] = {
    "variables": {
        "RESERVOIR_SIZE": [550] + [1100] * 6,  # unit: [m]
        "GRID_SIZE": ["20,5,5,5,5,5", 5, 10, 20, 5, 10, 20],
        "MLNEARWELLCONFIGFILE": [
            "",
            str(dirname / "nn" / "MLNearWellConfig.json"),
            str(dirname / "nn" / "MLNearWellConfig.json"),
            str(dirname / "nn" / "MLNearWellConfig.json"),
            "",
            "",
            "",
        ],
        "RUN_NAME": [
            "5x5m_Peaceman",
            "100x100m_NN",
            "52x52m_NN",
            "27x27m_NN",
            "100x100m_Peaceman",
            "52x52m_Peaceman",
            "27x27m_Peaceman",
        ],
        "USEMLNEARWELL": [
            "",
            "--UseMLNearWell=true",
            "--UseMLNearWell=true",
            "--UseMLNearWell=true",
            "",
            "",
            "",
        ],
    },
    "constants": {
        **runspecs_ensemble["constants"],
        **{
            "INIT_PRESSURE": 65 * units.BAR_TO_PASCAL,  # unit: [Pa]
            "INT_HEIGHT": 7.5,  # unit: [m]
            "PERM": 2e-13 * units.M2_TO_MILIDARCY,
            "RESERVOIR_SIZE": 1100,  # unit: [m]
            # Well radius is read from the radius of the innermost grid cell of the ensemble
            # simulation (~0.23) times the ``pyopmnearwell_correction`` factor (~1.1) to
            # translate from a triangle to a radial grid. Thus it differs from the ensemble
            # well radius.
            "WELL_RADIUS": 0.25,  # unit: [m]
            "OPM": OPM,
            "FLOW": FLOW,
        },
    },
}

runspecs_integration_2 = {
    **runspecs_integration_1,
    **{
        "constants": {
            **runspecs_integration_1["constants"],
            **{
                "INIT_PRESSURE": 80 * units.BAR_TO_PASCAL,
                "INT_HEIGHT": 15,
                "PERM": 5e-13 * units.M2_TO_MILIDARCY,
            },
        }
    },
}

runspecs_integration_3 = {
    **runspecs_integration_1,
    **{
        "constants": {
            **runspecs_integration_1["constants"],
            **{
                "INIT_PRESSURE": 90 * units.BAR_TO_PASCAL,
                "INT_HEIGHT": 20,
                "PERM": 5e-14 * units.M2_TO_MILIDARCY,
            },
        }
    },
}
