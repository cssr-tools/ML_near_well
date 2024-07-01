"""Note: The injection rate is fixed in the standardwell_impl.mako to calculate the
total injected gas. If changed in this file, the mako needs to be changed as well (line
2577).

Simulation parameters are loosely similar to the Utsira formation, taken from V. Singh,
A. Cavanagh, H. Hansen, B. Nazarian, M. Iding, and P. Ringrose, “Reservoir Modeling of
CO2 Plume Behavior Calibrated Against Monitoring Data From Sleipner, Norway”.


"""

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

NUM_MEMBERS: int = 250
# Surface density of CO2 - calculated with OPM PVT.
SURFACE_DENSITY: float = 1.86843  # unit [kg/m^3]

runspecs_ensemble: dict[str, Any] = {
    "npoints": NUM_MEMBERS,  # number of ensemble members
    "npruns": 5,  # number of parallel runs
    "variables": {
        # Assumed pressure regime of Utsira formation.
        "INIT_PRESSURE": (
            50 * units.BAR_TO_PASCAL,
            120 * units.BAR_TO_PASCAL,
            NUM_MEMBERS,
        ),  # unit: [Pa]
        # Permeability of Utsira formation is 1100 - 5000 mD ~ 1e-12 - 5e-12 m^2.
        "PERM": (
            5e-13 * units.M2_TO_MILIDARCY,
            1e-11 * units.M2_TO_MILIDARCY,
            NUM_MEMBERS,
        ),  # unit: [mD]
    },
    "constants": {
        # Seabed temperature of 7°C + 35°C/km at 800-1100m depth ~ 40°C
        "INIT_TEMPERATURE": 40,  # unit: [°C]
        # Porosity ot Utsira formation is 0.34 - 0.36
        "POROSITY": 0.35,  # unit: [-]
        "SURFACE_DENSITY": SURFACE_DENSITY,  # unit: [kg/m^3]
        # Surface rates for Utsira are 3e4 - 6e5 m^3/d on a perforation length of 38 m.
        # We take similar values on a perforation length of 5 m to obtain a slightly
        # larger pressure gradient.
        "INJECTION_RATE": 1e6 * SURFACE_DENSITY,  # unit: [kg/d]
        "INJECTION_TIME": 10,  # unit: [d]
        "REPORTSTEP_LENGTH": 0.1,  # unit [d]
        "NUM_XCELLS": 50,
        "LENGTH": 100,
        "HEIGHT": 5,  # unit: [m]
        # The well is fully inside the innermost cell, which we consider as the "well"
        # to ensure easier computation. The "well" radius for computation is thus
        # dependent on the cell size and this value gets disregarded when creating the
        # dataset.
        "WELL_RADIUS": 0.25,  # unit: [m]
        "FLOW": FLOW,
    },
}

trainspecs: dict[str, Any] = {
    "features": ["pressure", "geometr_WI", "V_tot"],
    "MinMax_scaling": True,
    "kerasify": True,
    "architecture": "fcnn",
    "permeability_log": False,
}

runspecs_integration_1: dict[str, Any] = {
    "variables": {
        "RESERVOIR_SIZE": [550] + [1100] * 6,  # unit: [m]
        "GRID_SIZE": ["20,5,5,5,5", 5, 10, 20, 5, 10, 20],
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
            "5x5m_Peaceman",
            "100x100m_NN",
            "52x52m_NN",
            "27x27m_NN",
            "100x100m_Peaceman",
            "52x52m_Peaceman",
            "27x27m_Peaceman",
        ],
    },
    "constants": runspecs_ensemble["constants"]
    | {
        "PERM": 1e-12 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "INIT_PRESSURE": 65 * units.BAR_TO_PASCAL,  # unit: [Pa]
        "OPM": OPM_ML,
        "FLOW": FLOW_ML,
        # Well radius is read from the radius of the innermost grid cell of the ensemble
        # calculation times the ``pyopmnearwell_correction`` factor to translate from a
        # triangle to a radial grid. Thus it differs from the ensemble well radius.
        "WELL_RADIUS": 0.35,  # unit: [m]
    },
}

runspecs_integration_2 = runspecs_integration_1 | {
    "constants": runspecs_integration_1["constants"]
    | {
        "INIT_PRESSURE": 90 * units.BAR_TO_PASCAL,
        "PERM": 5e-12 * units.M2_TO_MILIDARCY,
    }
}

runspecs_integration_3 = runspecs_integration_1 | {
    "constants": runspecs_integration_1["constants"]
    | {
        "INIT_PRESSURE": 80 * units.BAR_TO_PASCAL,
        "PERM": 9e-13 * units.M2_TO_MILIDARCY,
    }
}
