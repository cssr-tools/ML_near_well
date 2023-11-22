import pathlib
from typing import Any

import numpy as np
from pyopmnearwell.ml import ensemble, integration, nn
from pyopmnearwell.utils import units

dirname: pathlib.Path = pathlib.Path(__file__).parent

ensemble_dir: pathlib.Path = dirname / "ensemble"
data_dir: pathlib.Path = dirname / "dataset"
nn_dir: pathlib.Path = dirname / "nn"

ensemble_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
nn_dir.mkdir(parents=True, exist_ok=True)

OPM: pathlib.Path = pathlib.Path("/home/peter/Documents/2023_CEMRACS/opm")
FLOW: pathlib.Path = OPM / "build" / "opm-simulators" / "bin" / "flow"
OPM_ML: pathlib.Path = pathlib.Path("/home/peter/Documents/2023_CEMRACS/opm_ml")
FLOW_ML: pathlib.Path = (
    OPM_ML / "build" / "opm-simulators" / "bin" / "flow_gaswater_dissolution_diffuse"
)

# Run ensemble
runspecs_ensemble: dict[str, Any] = {
    "npoints": 300,  # number of ensemble members
    "npruns": 5,  # number of parallel runs
    "variables": {
        "INIT_PRESSURE": (50 * units.BAR_TO_PASCAL, 150 * units.BAR_TO_PASCAL, 300)
    },  # unit: [Pa]
    "constants": {
        "PERMX": 1e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "PERMZ": 1e-13 * units.M2_TO_MILIDARCY,  # unit: [mD]
        "INIT_TEMPERATURE": 40,  # unit: [°C]
        "SURFACE_DENSITY": 998.414,  # unit: [kg/m^3]
        "INJECTION_RATE": 6e1 * 998.414,  # unit: [kg/d]
        "INJECTION_RATE_PER_SECOND": 6e1
        * units.Q_per_day_to_Q_per_seconds,  # unit: [m^3/s]
        "WELL_RADIUS": 0.25,  # unit: [m]; Fixed during training.
        "FLOW": FLOW,
    },
}

h2o_ensemble = ensemble.create_ensemble(runspecs_ensemble)
ensemble.setup_ensemble(
    ensemble_dir,
    h2o_ensemble,
    (dirname / "h2o_ensemble.mako"),
    recalc_grid=False,
    recalc_sections=True,
    recalc_tables=False,
)
data: dict[str, Any] = ensemble.run_ensemble(
    FLOW, ensemble_dir, runspecs_ensemble, ["PRESSURE"], [], [], num_report_steps=100
)


# Create dataset
# Scale pressure to [Pa], since OPM uses [Pa] internally (in the ``METRIC`` mode) i.e.,
# the input to the neural network will be in [Pa].
pressures: np.ndarray = ensemble.extract_features(
    data, keywords=["PRESSURE"], keyword_scalings={"PRESSURE": units.BAR_TO_PASCAL}
)

# Truncate outer and inner cells. Truncate every time step but the last one.
# The innermost radius corresponds to the bottom hole pressure and is already truncated
# for the ``WI`` array.
pressures = pressures[..., -1, 5:-4, :]
radiis: np.ndarray = ensemble.calculate_radii(
    (ensemble_dir / "runfiles_0" / "preprocessing" / "GRID.INC")
)
radiis = radiis[5:-4]
assert pressures.shape[-2] == radiis.shape[-1]

WI: np.ndarray = ensemble.calculate_WI(data, runspecs_ensemble, num_zcells=1)[0][
    ..., -1, 4:-4
]
ensemble.store_dataset(
    np.stack(
        [
            pressures.flatten(),
            np.tile(radiis, np.prod(pressures.shape[:-2]).item()),
        ],
        axis=-1,
    ),
    WI.flatten()[..., None],
    data_dir,
)

# Train model
model = nn.get_FCNN(ninputs=2, noutputs=1)
train_data, val_data = nn.scale_and_prepare_dataset(
    data_dir, ["pressure", "radius"], nn_dir
)
nn.train(model, train_data, val_data, savepath=nn_dir, epochs=100)

# Integrate
runspecs_integration: dict[str, Any] = {
    "variables": {
        "GRID_SIZE": [20, 20, 100],
        "ML_MODEL_PATH": [str((dirname / "nn" / "WI.model")), "", ""],
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
integration.recompile_flow((nn_dir / "scalings.csv"), OPM_ML, "h2o_2_inputs")
integration.run_integration(
    runspecs_integration, (dirname / "integration"), (dirname / "h2o_integration.mako")
)
