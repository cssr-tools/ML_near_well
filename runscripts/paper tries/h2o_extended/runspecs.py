import pathlib
from typing import Any

from pyopmnearwell.utils import units

dirname: pathlib.Path = pathlib.Path(__file__).parent

OPM: pathlib.Path = pathlib.Path("/home/peter/Documents/2023_CEMRACS/opm")
FLOW: pathlib.Path = OPM / "build" / "opm-simulators" / "bin" / "flow"
OPM_ML: pathlib.Path = pathlib.Path("/home/peter/Documents/2023_CEMRACS/opm_ml")
FLOW_ML: pathlib.Path = (
    OPM_ML / "build" / "opm-simulators" / "bin" / "flow_gaswater_dissolution_diffuse"
)

NUM_MEMBERS: int = 200
SURFACE_DENSITY: float = 998.414

runspecs_ensemble: dict[str, Any] = {
    "npoints": NUM_MEMBERS,  # number of ensemble members
    "npruns": 5,  # number of parallel runs
    "variables": {
        "INIT_PRESSURE": (
            50 * units.BAR_TO_PASCAL,
            150 * units.BAR_TO_PASCAL,
            NUM_MEMBERS,
        ),  # unit: [Pa]
        "INT_HEIGHT": (1, 30, NUM_MEMBERS),  # unit: [m]
        "PERM": (
            1e-14 * units.M2_TO_MILIDARCY,
            1e-12 * units.M2_TO_MILIDARCY,
            NUM_MEMBERS,
        ),  # unit: [mD]
    },
    "constants": {
        "INIT_TEMPERATURE": 40,  # unit: [°C]
        "POROSITY": 0.25,  # unit: [-]
        "SURFACE_DENSITY": SURFACE_DENSITY,  # unit: [kg/m^3]
        "INJECTION_RATE": 6e1 * SURFACE_DENSITY,  # unit: [kg/d]
        "INJECTION_TIME": 10,  # unit: [d]
        "REPORTSTEP_LENGTH": 0.1,  # unit [d]
        "NUM_XCELLS": 50,
        "LENGTH": 100,
        "WELL_RADIUS": 0.25,  # unit: [m]
        "FLOW": FLOW,
    },
}

trainspecs: dict[str, Any] = {
    "features": ["pressure", "permeability", "height", "radius"],
    "MinMax_scaling": True,
    "kerasify": True,
    "architecture": "fcnn",
    "permeability_log": False,
}

runspecs_integration: dict[str, Any] = {
    "variables": {
        "GRID_SIZE": [20, 20, 100],
        "ML_MODEL_PATH": [str((dirname / "nn" / "WI.model")), "", ""],
        "RUN_NAME": ["125x125m_NN", "125x125m_Peaceman", "25x25m_Peaceman"],
    },
    "constants": runspecs_ensemble["constants"]
    + {
        "PERM": 1e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "INT_HEIGHT": 5,  # unit: [m]
        "INIT_PRESSURE": 65 * units.BAR_TO_PASCAL,  # unit: [Pa]
        "RESERVOIR_SIZE": 5000,  # unit: [m]
    },
}
runspecs_integration["constants"].update("FLOW", FLOW_ML)
