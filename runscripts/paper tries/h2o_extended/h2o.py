import pathlib
import sys
from typing import Any

import numpy as np
import seaborn as sns
from pyopmnearwell.ml import ensemble, integration, nn
from pyopmnearwell.utils import units
from runspecs import runspecs_ensemble, runspecs_integration, trainspecs
from tensorflow import keras

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
    data: np.ndarray = full_ensemble(
        runspecs_ensemble,
        ensemble_dir,
        ecl_keywords=["PRESSURE", "FLOWATI+"],
        init_keywords=["PERMX", "DZ"],
        recalc_grid=False,
        recalc_sections=True,
        recalc_tables=False,
        keyword_scalings={
            # Scale pressure to [Pa], since OPM uses [Pa] internally (in the ``METRIC``
            # mode) i.e., the input to the neural network will be in [Pa].
            "PRESSURE": units.BAR_TO_PASCAL,  #
            "PERMX": units.MILIDARCY_TO_M2,
        },
    )  # ``shape=(num_members, num_report_steps, num_zcells, num_xcells, 4)``
    np.save(str(ensemble_dir / "data"), data)


# Create dataset
if True:
    data = np.load(str(ensemble_dir / "data"))
    radii: np.ndarray = ensemble.calculate_radii(  # type: ignore
        (ensemble_dir / "runfiles_0" / "preprocessing" / "GRID.INC")
    )
    WI: np.ndarray = ensemble.calculate_WI(data[..., 0], data[..., 1])[0]

    # Truncate outer and inner cells. Truncate every time step but the last one.
    # The innermost radius corresponds to the bottom hole pressure and is already truncated
    # for the ``WI`` array.
    pressures: np.ndarray = data[..., -1, 1:-1, 0]
    permeabilities: np.ndarray = data[..., -1, 1:-1, 2]
    heights: np.ndarray = data[..., -1, 1:-1, 3]
    radii = radii[1:-1]
    WI = WI[..., :-1]

    assert pressures.shape[-1] == radii.shape[-1]
    assert pressures.shape == permeabilities.shape
    assert pressures.shape == heights.shape
    assert pressures.shape == WI.shape

    features: np.ndarray = np.stack(
        [
            pressures.flatten(),
            permeabilities.flatten(),
            heights.flatten(),
            np.broadcast_to(radii, pressures.shape).flatten(),
        ],
        axis=-1,
    )
    targets: np.ndarray = WI.flatten()[..., None]

    ensemble.store_dataset(
        np.stack(
            [
                pressures.flatten(),
                permeabilities.flatten(),
                heights.flatten(),
                np.broadcast_to(radii, pressures.shape).flatten(),
            ],
            axis=-1,
        ),
        WI.flatten()[..., None],
        data_dir,
    )

    # Plot some WI vs radius and vs time.
    for i in range(0, runspecs_ensemble["npoints"], 30):
        plot_member(
            features,
            targets,
            i,
            data_dir / f"member_{i}_WI_vs_radius",
            comparison_param="layer",
            fixed_param_index=10,  # Plot for time step 10.
            radius_index=3,
            permeability_index=1,
        )
        plot_member(
            features,
            targets,
            i,
            data_dir / f"member_{i}_WI_vs_time",
            x_param="time",
            comparison_param="layer",
            final_time=runspecs_ensemble["constants"]["INJECTION_TIME"],
            fixed_param_index=10,  # Plot for time step 10.
            radius_index=3,
            permeability_index=1,
        )

# Train model
if True:
    train_data, val_data = nn.scale_and_prepare_dataset(  # type: ignore
        data_dir, ["pressure", "permeability", "height", "radius"], nn_dir
    )

    tune_and_train(trainspecs, data_dir, nn_dir, max_trials=1, lr=1e-3)
    model: keras.Model = keras.models.load_model(nn_dir / "bestmodel.keras")  # type: ignore
    for i in range(0, runspecs_ensemble["npoints"], 30):
        # Plot some NN WI and data WI vs radius and vs time.
        plot_member(
            features,
            targets,
            i,
            nn_dir / f"member_{i}_WI_vs_radius",
            comparison_param="layer",
            fixed_param_index=10,  # Plot for time step 10.
            radius_index=3,
            permeability_index=1,
            model=model,
            nn_dirname=nn_dir,
            trainspecs=trainspecs,
        )
        plot_member(
            features,
            targets,
            i,
            nn_dir / f"member_{i}_WI_vs_time",
            x_param="time",
            comparison_param="layer",
            final_time=runspecs_ensemble["constants"]["INJECTION_TIME"],
            fixed_param_index=10,  # Plot for time step 10.
            radius_index=3,
            permeability_index=1,
            model=model,
            nn_dirname=nn_dir,
            trainspecs=trainspecs,
        )

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
