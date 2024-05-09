import math
import pathlib
import sys
from typing import Any

import numpy as np
import seaborn as sns
from pyopmnearwell.ml import ensemble, integration, nn
from pyopmnearwell.utils import formulas, units
from runspecs import runspecs_ensemble, runspecs_integration, trainspecs
from tensorflow import keras

dirname: pathlib.Path = pathlib.Path(__file__).parent

# Have to import like this.
sys.path.append(str(dirname / ".."))
from utils import full_ensemble, plot_member, read_and_plot_bhp, tune_and_train

# Set seaborn style.
sns.set_theme(context="paper", style="whitegrid")


ensemble_dir: pathlib.Path = dirname / "ensemble"
data_dir: pathlib.Path = dirname / "dataset"
nn_dir: pathlib.Path = dirname / "nn"

ensemble_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
nn_dir.mkdir(parents=True, exist_ok=True)


# Run ensemble and extract data
if False:
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
    # Truncate all but the last time step since the problem is steady state.
    np.save(str(ensemble_dir / "data"), pressures[:, -1, ...])

# Create dataset
if False:
    pressures = np.load(str(ensemble_dir / "data.npy"))
    # Get radii and transform from triangle grid to cake grid.
    radii: np.ndarray = ensemble.calculate_radii(  # type: ignore
        (ensemble_dir / "runfiles_0" / "preprocessing" / "GRID.INC"),
        # For some reason only ``NUM_XCELLS - 1`` cells are generated.
        num_cells=runspecs_ensemble["constants"]["NUM_XCELLS"] - 1,
    ) * formulas.pyopmnearwell_correction(2 * math.pi / 6)
    # Note that the injection rate is specified for a full 360° grid. As this is the
    # same in the full simulation the model will be employed in, we keep it that way.
    WI: np.ndarray = ensemble.calculate_WI(
        pressures[..., 0], runspecs_ensemble["constants"]["INJECTION_RATE"]
    )[0]

    # Truncate outer and inner cells.
    # The innermost radius corresponds to the bottom hole pressure and is already
    # truncated for the ``WI`` array.
    pressures = pressures[..., 1:-1, 0]
    radii = radii[1:-1]
    WI = WI[..., :-1]
    assert pressures.shape[-1] == radii.shape[-1]
    assert pressures.shape == WI.shape

    features: np.ndarray = np.stack(np.broadcast_arrays(pressures, radii), axis=-1)
    targets: np.ndarray = WI

    ensemble.store_dataset(features.reshape(-1, 2), targets.reshape(-1, 1), data_dir)

    # Plot some WI vs radius.
    for i in range(0, runspecs_ensemble["npoints"], 30):
        plot_member(
            # Add a time and z axis for plotting.
            features[:, None, None, ...],
            targets[:, None, None, ...],
            i,
            data_dir / f"member_{i}_WI_vs_radius",
            comparison_param="layer",
            radius_index=1,
        )

# Train model
if False:
    tune_and_train(trainspecs, data_dir, nn_dir, max_trials=10, lr=1e-3)
    model: keras.Model = keras.models.load_model(nn_dir / "bestmodel.keras")  # type: ignore
    # Plot NN WI and data WI vs radius.
    plot_member(
        features[:, None, None, ...],
        targets[:, None, None, ...],
        i,
        nn_dir / f"member_{i}_WI_vs_radius",
        comparison_param="layer",
        radius_index=1,
        model=model,
        nn_dirname=nn_dir,
        trainspecs=trainspecs,
    )

# Integrate into OPM.
if False:
    integration.recompile_flow(
        (nn_dir / "scalings.csv"),
        runspecs_integration["constants"]["OPM"],
        dirname / "standardwell_impl.mako",
        dirname / "standardwell.hpp",
    )
    integration.run_integration(
        runspecs_integration,
        (dirname / "integration"),
        (dirname / "integration.mako"),
    )

# Plot results.
if False:
    summary_files: list[pathlib.Path] = [
        dirname / "integration" / "run_6" / "output" / "12.5X12.5M_PEACEMAN.SMSPEC",
        dirname / "integration" / "run_0" / "output" / "125X125M_NN.SMSPEC",
        dirname / "integration" / "run_2" / "output" / "62x62M_NN.SMSPEC",
        dirname / "integration" / "run_4" / "output" / "25X25M_NN.SMSPEC",
        dirname / "integration" / "run_1" / "output" / "125X125M_PEACEMAN.SMSPEC",
        dirname / "integration" / "run_3" / "output" / "62x62M_PEACEMAN.SMSPEC",
        dirname / "integration" / "run_5" / "output" / "25X25M_PEACEMAN.SMSPEC",
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
        + list(plt.cm.Blues(np.linspace(0.7, 0.3, 3)))
        + list(plt.cm.Greys(np.linspace(0.7, 0.3, 3)))
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
    read_and_plot_bhp(summary_files, labels, colors, linestyles, dirname / "bhp.svg")


# Plot results
if False:
    summary_files: list[pathlib.Path] = [
        dirname / "integration" / "run_2" / "output" / "25X25M_PEACEMAN.SMSPEC",
        dirname / "integration" / "run_0" / "output" / "125X125M_NN.SMSPEC",
        dirname / "integration" / "run_1" / "output" / "125X125M_PEACEMAN.SMSPEC",
    ]
    labels: list[str] = [
        "Fine-scale benchmark",
        "125x125m NN",
        "125x125m Peaceman",
    ]
    colors: list[str] = (
        ["black"]
        + list(plt.cm.Blues(np.linspace(0.7, 0.3, 1)))
        + list(plt.cm.Greys(np.linspace(0.7, 0.3, 1)))
    )
    linestyles: list[str] = [
        "solid",
        "dashed",
        "dotted",
    ]
    read_and_plot_bhp(summary_files, labels, colors, linestyles, dirname / "bhp.svg")