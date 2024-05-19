import os
from typing import Any

import numpy as np

from pyopmnearwell.ml import ensemble
from pyopmnearwell.utils import units

dirname: str = os.path.dirname(__file__)
os.makedirs(os.path.join(dirname, "ensemble"), exist_ok=True)
os.makedirs(os.path.join(dirname, "dataset"), exist_ok=True)
os.makedirs(os.path.join(dirname, "nn"), exist_ok=True)
os.makedirs(os.path.join(dirname, "integration"), exist_ok=True)
OPM: str = "/home/peter/Documents/2023_CEMRACS/opm"
FLOW: str = f"{OPM}/build/opm-simulators/bin/flow"
OPM_ML: str = "/home/peter/Documents/2023_CEMRACS/opm_ml"
FLOW_ML: str = f"{OPM_ML}/build/opm-simulators/bin/flow_gaswater_dissolution_diffuse"

# Run ensemble
# Create permeability fields:

num_layers: int = 5
height: float = 20.0

variables: dict[str, tuple[float, float, int]] = {
    f"PERMX_{i}": (  # unit: [mD]
        1e-15 * units.M2_TO_MILIDARCY,
        1e-9 * units.M2_TO_MILIDARCY,
        40,
    )
    for i in range(num_layers)
}

# Vertical permeabilties correspond to horizontal permeability.
variables.update(
    {
        f"PERMZ_by_{key}": (  # unit: [mD]
            1,
            1,
            npoints,
        )
        for key, (_, __, npoints) in variables.items()
    }
)

variables.update(
    {
        "INIT_PRESSURE": (
            50 * units.BAR_TO_PASCAL,
            80 * units.BAR_TO_PASCAL,
            30,
        ),  # unit: [Pa]
        "INIT_TEMPERATURE": (20, 40, 20),  # unit: [°C])
    }
)

runspecs_ensemble: dict[str, Any] = {
    "npoints": 5,  # number of ensemble members
    "npruns": 5,  # number of parallel runs
    "variables": variables,
    "constants": {
        "SURFACE_DENSITY": 1.86843,  # unit: [kg/m^3]
        "INJECTION_RATE": 5e5 * 1.86843,  # unit: [kg/d]
        "INJECTION_RATE_PER_SECOND": 5e5
        * units.Q_per_day_to_Q_per_seconds,  # unit: [m^3/s]
        "WELL_RADIUS": 0.25,  # unit: [m]; Fixed during training.
        "FLOW": FLOW,
        "NUM_LAYERS": num_layers,
        "HEIGHT": height,
        "THICKNESS": height / num_layers,
    },
}

co2_ensemble = ensemble.create_ensemble(
    runspecs_ensemble,
    efficient_sampling=[f"PERMX_{i}" for i in range(num_layers)]
    + [f"PERMZ_by_PERMX_{i}" for i in range(num_layers)],
)
ensemble.setup_ensemble(
    os.path.join(dirname, "ensemble"),
    co2_ensemble,
    os.path.join(dirname, "co2_ensemble.mako"),
    recalc_grid=False,
    recalc_sections=True,
    recalc_tables=False,
)
data: dict[str, Any] = ensemble.run_ensemble(
    FLOW,
    os.path.join(dirname, "ensemble"),
    runspecs_ensemble,
    ecl_keywords=["PRESSURE", "SGAS"],
    init_keywords=["PERMX", "PERMZ"],
    summary_keywords=[],
    num_report_steps=100,
)


# Get dataset
# Get only data from the cells at 25 m distance from the well. Take only every 3rd time step.
features: np.ndarray = np.array(
    ensemble.extract_features(
        data,
        keywords=["PRESSURE", "SGAS", "PERMX", "PERMZ"],
        keyword_scalings={"PRESSURE": units.BAR_TO_PASCAL},
        # TODO: OPM handles permeability in [mD] -> Make sure this is correct.
    )
)[..., ::3, 100::400, :]

timesteps: np.ndarray = np.arange(features.shape[-3])  # No unit.

WI: np.ndarray = ensemble.calculate_WI(data, runspecs_ensemble, num_zcells=num_layers)[
    ..., ::3, 100::400
]

# Features are, in the following order:
# 1. PRESSURE - cell
# 2. SGAS - cell
# 3. PERMX - cell
# 4. PERMZ - cell
# 5. TIME


ensemble.store_dataset(
    np.stack(
        [
            features[..., 0],
            features[..., 1],
            features[..., 2],
            features[..., 3],
            np.broadcast_to(timesteps[..., None], features[..., 0].shape),
        ],
        axis=-1,
    ),
    WI[..., None],
    os.path.join(dirname, "dataset"),
)
