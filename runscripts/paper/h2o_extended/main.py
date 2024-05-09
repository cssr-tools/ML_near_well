import math
import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyopmnearwell.ml import analysis, ensemble, integration, nn, utils
from pyopmnearwell.utils import formulas, units
from runspecs import runspecs_ensemble, runspecs_integration, trainspecs
from tensorflow import keras

dirname: pathlib.Path = pathlib.Path(__file__).parent

# Have to import like this.
sys.path.append(str(dirname / ".."))
from utils import full_ensemble, plot_member, read_and_plot_bhp, tune_and_train

# Set seaborn style.
sns.set_theme(context="paper", style="whitegrid")

SEED: int = 19123
utils.enable_determinism(SEED)

ensemble_dir: pathlib.Path = dirname / "ensemble"
data_dir: pathlib.Path = dirname / "dataset"
nn_dir: pathlib.Path = dirname / "nn"
integration_dir: pathlib.Path = dirname / "integration"

ensemble_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
nn_dir.mkdir(parents=True, exist_ok=True)
integration_dir.mkdir(parents=True, exist_ok=True)

# Run ensemble and extract data.
if False:
    data: np.ndarray = full_ensemble(
        runspecs_ensemble,
        ensemble_dir,
        ecl_keywords=["PRESSURE", "FLOWATI+"],
        init_keywords=["PERMX", "DZ"],
        recalc_grid=True,
        recalc_sections=True,
        recalc_tables=False,
        keyword_scalings={
            # Scale pressure to [Pa], since OPM uses [Pa] internally (in the ``METRIC``
            # mode) i.e., the input to the neural network will be in [Pa].
            "PRESSURE": units.BAR_TO_PASCAL,
            # Scale permeability to [m^2], since OPM uses [m^2] internally (in the
            # ``METRIC`` mode) i.e., the input to the neural network will be in [m^2].
            "PERMX": units.MILIDARCY_TO_M2,
        },
        seed=SEED,
    )  # ``shape=(num_members, num_report_steps, num_zcells, num_xcells, 4)``
    # Truncate all but the last time step since the problem is steady state.
    np.save(str(ensemble_dir / "data"), data[:, -1, ...])


# Create dataset.
if True:
    # Truncate the outermost cell already here to avoid getting nan in the WIs.
    data = np.load(str(ensemble_dir / "data.npy"))[..., :-1, :]
    # Get radii and transform from triangle grid to cake grid.
    radii: np.ndarray = ensemble.calculate_radii(  # type: ignore
        (ensemble_dir / "runfiles_0" / "preprocessing" / "GRID.INC"),
        # For some reason only ``NUM_XCELLS - 1`` cells are generated.
        num_cells=runspecs_ensemble["constants"]["NUM_XCELLS"] - 1,
    ) * formulas.pyopmnearwell_correction(2 * math.pi / 6)
    # Injection rate is flow rate at zeroth x cell. To use it in coarse scale
    # simulations in OPM, convert to convert to the right radial angle (ensemble
    # simulation run only on 60° instead of 360°) and to per seconds.
    WI: np.ndarray = ensemble.calculate_WI(
        data[..., 0], data[..., 0, 1] * 6 * units.Q_per_day_to_Q_per_seconds
    )[0]

    # Truncate well cells. Truncate every time step but the last one.
    # The innermost radius corresponds to the bottom hole pressure and is already
    # truncated for the ``WI`` array.
    pressures: np.ndarray = data[..., 1:, 0]
    permeabilities: np.ndarray = data[..., 1:, 2]
    heights: np.ndarray = data[..., 1:, 3]
    radii = radii[1:-1]

    assert (
        pressures.shape[-1] == radii.shape[-1]
    ), "``radii.shape`` does not equal ``pressure.shape"
    assert (
        pressures.shape == permeabilities.shape
    ), "``permeabilities.shape`` does not equal ``pressure.shape"
    assert (
        pressures.shape == heights.shape
    ), "``heights.shape`` does not equal ``pressure.shape"
    assert pressures.shape == WI.shape, "``WI.shape`` does not equal ``pressure.shape"

    features: np.ndarray = np.stack(
        np.broadcast_arrays(pressures, permeabilities, heights, radii), axis=-1
    )
    targets: np.ndarray = WI

    ensemble.store_dataset(features.reshape(-1, 4), targets.reshape(-1, 1), data_dir)

    # Plot some WI vs radius.
    for i in range(0, runspecs_ensemble["npoints"], 30):
        plot_member(
            # Add a time and z axis for plotting.
            features[:, None, None, ...],
            targets[:, None, None, ...],
            i,
            data_dir / f"member_{i}_WI_vs_radius",
            comparison_param="layer",
            radius_index=3,
            permeability_index=1,
        )

# Train model.
if True:
    tune_and_train(
        trainspecs,
        data_dir,
        nn_dir,
        max_trials=20,
        lr=1e-4,
        epochs=5000,
        executions_per_trial=2,
    )

# Do some plotting of results and analysis.
if True:
    model: keras.Model = keras.models.load_model(nn_dir / "bestmodel.keras")  # type: ignore
    for i in range(0, runspecs_ensemble["npoints"], 30):
        # Plot NN WI and data WI vs radius.
        plot_member(
            features[:, None, None, ...],
            targets[:, None, None, ...],
            i,
            nn_dir / f"member_{i}_WI_vs_radius",
            comparison_param="layer",
            radius_index=3,
            permeability_index=1,
            model=model,
            nn_dirname=nn_dir,
            trainspecs=trainspecs,
        )
    outputs, inputs = analysis.sensitivity_analysis(model)
    analysis.plot_analysis(
        outputs,
        inputs,
        nn_dir / "sensitivity_analysis",
        feature_names=[
            r"$p [\mathrm{bar}]$",
            r"$k [\mathrm{m}^2]$",
            r"$h [\mathrm{m}]$",
            r"$r_e [\mathrm{m}]$",
        ],
    )

# Integrate into OPM.
if True:
    integration.recompile_flow(
        (nn_dir / "scalings.csv"),
        runspecs_integration["constants"]["OPM"],
        dirname / "standardwell_impl.mako",
        dirname / "standardwell.hpp",
    )
    integration.run_integration(
        runspecs_integration,
        integration_dir,
        (dirname / "integration.mako"),
    )

# Plot results.
if True:
    summary_files: list[pathlib.Path] = [
        integration_dir / "run_6" / "output" / "5X5M_PEACEMAN.SMSPEC",
        integration_dir / "run_0" / "output" / "100X100M_NN.SMSPEC",
        integration_dir / "run_2" / "output" / "43X43M_NN.SMSPEC",
        integration_dir / "run_4" / "output" / "21X21M_NN.SMSPEC",
        integration_dir / "run_1" / "output" / "100X100M_PEACEMAN.SMSPEC",
        integration_dir / "run_3" / "output" / "43X43M_PEACEMAN.SMSPEC",
        integration_dir / "run_5" / "output" / "21X21M_PEACEMAN.SMSPEC",
    ]
    labels: list[str] = [
        "Fine-scale benchmark",
        "125x125m NN",
        "62.5x62.5m NN",
        "25x25m NN",
        "125x125m Peaceman",
        "62.5x62.5m Peaceman",
        "25x25m Peaceman",
    ]
    colors: list[str] = (
        ["black"]
        + list(plt.cm.Blues(np.linspace(0.7, 0.3, 3)))  # type: ignore
        + list(plt.cm.Greys(np.linspace(0.7, 0.3, 3)))  # type: ignore
    )
    linestyles: list[str] = [
        "solid",
        "dashed",
        "dashed",
        "dashed",
        "dotted",
        "dotted",
        "dotted",
    ]
    read_and_plot_bhp(
        summary_files, labels, colors, linestyles, integration_dir / "bhp.svg"
    )
