import pathlib
import sys
from typing import Any

import numpy as np
import seaborn as sns
from pyopmnearwell.ml import ensemble, integration, nn
from pyopmnearwell.utils import units
from runspecs import runspecs_ensemble, runspecs_integration, trainspecs

dirname: pathlib.Path = pathlib.Path(__file__).parent

# Have to import like this.
sys.path.append(str(dirname / ".."))
from utils import full_ensemble, plot_member, tune_and_train

# Set seaborn style.
sns.set_theme(context="paper", style="whitegrid")


ensemble_dir: pathlib.Path = dirname / "ensemble"
data_dir: pathlib.Path = dirname / "dataset"
nn_dir: pathlib.Path = dirname / "nn"

ensemble_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
nn_dir.mkdir(parents=True, exist_ok=True)


# Run ensemble and extract data
if True:
    pressures: np.ndarray = full_ensemble(
        runspecs_ensemble,
        ensemble_dir,
        ecl_keywords=["PRESSURE"],
        recalc_grid=False,
        recalc_sections=True,
        recalc_tables=False,
        keyword_scalings={
            # Scale pressure to [Pa], since OPM uses [Pa] internally (in the ``METRIC``
            # mode) i.e., the input to the neural network will be in [Pa].
            "PRESSURE": units.BAR_TO_PASCAL,
        },
    )  # ``shape=(num_members, num_report_steps, num_zcells, num_xcells, 1)``
    np.save(str(ensemble_dir / "data"), pressures)

# Create dataset
if True:
    pressures = np.load(str(ensemble_dir / "data"))
    # Create dataset
    radii: np.ndarray = ensemble.calculate_radii(  # type: ignore
        (ensemble_dir / "runfiles_0" / "preprocessing" / "GRID.INC")
    )
    WI: np.ndarray = ensemble.calculate_WI(
        pressures[..., 0], runspecs_ensemble["constant"]["INJECTION_RATE"]
    )[0]

    # Truncate outer and inner cells. Truncate every time step but the last one.
    # The innermost radius corresponds to the bottom hole pressure and is already truncated
    # for the ``WI`` array.
    pressures = pressures[..., -1, 1:-1, 0]
    radii = radii[1:-1]
    WI = WI[..., :-1]
    assert pressures.shape[-1] == radii.shape[-1]
    assert pressures.shape == WI.shape

    ensemble.store_dataset(
        np.stack(
            [
                pressures.flatten(),
                np.broadcast_to(radii, pressures.shape).flatten(),
                # np.tile(radiis, np.prod(pressures.shape[:-2]).item()),
            ],
            axis=-1,
        ),
        WI.flatten()[..., None],
        data_dir,
    )

# Train model
if True:
    model = nn.get_FCNN(ninputs=2, noutputs=1)
    train_data, val_data = nn.scale_and_prepare_dataset(  # type: ignore
        data_dir, ["pressure", "radius"], nn_dir
    )
    tune_and_train(trainspecs, data_dir, nn_dir, max_trials=1, lr=1e-3)
    model: keras.Model = keras.models.load_model(nn_dir / "bestmodel.keras")  # type: ignore


# Integrate
if True:
    integration.recompile_flow(
        (nn_dir / "scalings.csv"),
        runspecs_integration["constants"]["OPM"],
        "h2o_2_inputs",
    )
    integration.run_integration(
        runspecs_integration,
        (dirname / "integration"),
        (dirname / "h2o_integration.mako"),
    )
