import csv
import os
from typing import Any

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from pyopmnearwell.ml import ensemble, integration, nn
from pyopmnearwell.utils import formulas, units

height: float = 50
num_zcells: int = 10
npoints: int = 5
INJECTION_RATE_PER_SECOND: float = (
    5e4 * 6 * units.Q_per_day_to_Q_per_seconds
)  # unit: [m^3/s]
INIT_TEMPERATURE: float = 25  # unit: [°C])
PERMX: float = 1e-12 * units.M2_TO_MILIDARCY  # unit: [mD]
SURFACE_DENSITY: float = 1.86843  # unit: [kg/m^3]
timesteps: np.ndarray = np.arange(34)  # No unit.

dirname: str = os.path.dirname(__file__)
OPM_ML: str = "/home/peter/Documents/2023_CEMRACS/opm_ml"
FLOW_ML: str = f"{OPM_ML}/build/opm-simulators/bin/flow_gaswater_dissolution_diffuse"

# Integrate
runspecs_integration: dict[str, Any] = {
    "variables": {
        "GRID_SIZE": [20, 20, 100],
        "ML_MODEL_PATH": [os.path.join(dirname, "nn", "WI.model"), "", ""],
        "RUN_NAME": ["125x125m_NN", "125x125m_Peaceman", "25x25m_Peaceman"],
    },
    "constants": {
        "PERMX": 1e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERMZ": 1e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "INIT_PRESSURE": 65 * units.BAR_TO_PASCAL,  # unit: [Pa]
        "INIT_TEMPERATURE": 40,  # unit: [°C]
        "INJECTION_RATE": 6e1 * 998.414,  # unit: [kg/d]
        "WELL_RADIUS": 0.25,  # unit: [m]; Fixed during training.
        "RESERVOIR_SIZE": 5000,  # unit: [m]
        "FLOW": FLOW_ML,
    },
}
integration.recompile_flow(
    os.path.join(dirname, "nn", "scalings.csv"),
    "co2_local_stencil",
    OPM_ML,
    standard_well_file="local_stencil",
    stencil_size=3,
    cell_feature_names=["pressure", "saturation"],
    num_cell_features=2,  # pressure and saturation
)
integration.run_integration(
    runspecs_integration,
    os.path.join(dirname, "integration"),
    os.path.join(dirname, "h2o_integration.mako"),
)
