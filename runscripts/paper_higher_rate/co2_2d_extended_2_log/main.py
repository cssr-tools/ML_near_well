import math
import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyopmnearwell.ml import analysis, ensemble, integration, nn, utils
from pyopmnearwell.utils import formulas, units
from runspecs import (
    runspecs_ensemble,
    runspecs_integration_1,
    runspecs_integration_2,
    runspecs_integration_3,
    trainspecs,
)
from tensorflow import keras
from upscale import CO2_2D_Upscaler

dirname: pathlib.Path = pathlib.Path(__file__).parent

sys.path.append(str(dirname / ".."))
from utils import full_ensemble, plot_member, read_and_plot_bhp, tune_and_train

# Set seaborn style.
sns.set_theme(context="paper", style="whitegrid")

SEED: int = 19123
utils.enable_determinism(SEED)

# Structure directories.
ensemble_dir: pathlib.Path = dirname / "ensemble"
data_dir: pathlib.Path = dirname / "dataset"
nn_dir: pathlib.Path = dirname / "nn"
integration_dir_1: pathlib.Path = dirname / "integration_1"
integration_dir_2: pathlib.Path = dirname / "integration_2"
integration_dir_3: pathlib.Path = dirname / "integration_3"

ensemble_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
nn_dir.mkdir(parents=True, exist_ok=True)
integration_dir_1.mkdir(parents=True, exist_ok=True)
integration_dir_2.mkdir(parents=True, exist_ok=True)
integration_dir_3.mkdir(parents=True, exist_ok=True)

# Get OPM installations.
# TODO: These need to be adjusted for reproducing results.
OPM: pathlib.Path = pathlib.Path("/home/peter/Documents/2023_CEMRACS/opm")
FLOW: pathlib.Path = OPM / "build" / "opm-simulators" / "bin" / "flow"
OPM_ML: pathlib.Path = pathlib.Path("/home/peter/Documents/2023_CEMRACS/opm_ml")
FLOW_ML: pathlib.Path = (
    OPM_ML / "build" / "opm-simulators" / "bin" / "flow_gaswater_dissolution_diffuse"
)


# Run ensemble and extract data.
if False:
    extracted_data: np.ndarray = full_ensemble(
        runspecs_ensemble,
        ensemble_dir,
        ecl_keywords=["PRESSURE", "FLOGASI+"],
        init_keywords=["PERMX", "DZ"],
        summary_keywords=["FGIT"],
        recalc_grid=False,
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
    )
    np.save(str(ensemble_dir / "features"), extracted_data)

# Upscale and create dataset.
if False:
    extracted_data = np.load(str(ensemble_dir / "features.npy"))
    upscaler = CO2_2D_Upscaler(
        extracted_data,
        runspecs_ensemble,
        data_dim=5,  # Dimension of a single datapoint.
    )
    features, targets = upscaler.create_ds(
        ensemble_dir, step_size_x=5, step_size_t=2, log_WI=True, log_geom_WI=True
    )

    # Remove WI_analytical for training.
    if False:
        train_features: np.ndarray = features[..., :-1]
    # Flatten features and targets before storing.
    ensemble.store_dataset(
        np.reshape(features, (-1, len(trainspecs["features"]))),
        np.reshape(targets, (-1, 1)),
        data_dir,
    )

# Plot some WIs.
# NOTE: The plots will have wrong x-axis labels and plot labels, as neither :math:`r_e`
# nor :math:`k` are present in the data (they are replaced by the geometrical part of
# the well index).
if False:
    for i in range(0, features.shape[0], 50):
        # Plot data WI vs radius.
        plot_member(
            features,
            targets,
            i,
            data_dir / f"member_{i}_WI_vs_radius",
            comparison_param="layer",
            fixed_param_index=10,  # Plot for time step 10.
            radius_index=1,
            permeability_index=1,
        )
        # Plot data WI vs time.
        plot_member(
            features,
            targets,
            i,
            data_dir / f"member_{i}_WI_vs_time",
            x_param="time",
            comparison_param="layer",
            final_time=runspecs_ensemble["constants"]["INJECTION_TIME"],
            fixed_param_index=5,  # Plot for radius 5.
            radius_index=1,
            permeability_index=1,
        )

# Train model and do some plotting of results and analysis.
if False:
    tune_and_train(
        trainspecs,
        data_dir,
        nn_dir,
        max_trials=10,
        lr=5e-4,
        epochs=5000,
        executions_per_trial=1,
    )
    model: keras.Model = keras.models.load_model(nn_dir / "bestmodel.keras")  # type: ignore
    for i in range(0, features.shape[0], 30):
        # Plot NN WI and data WI vs radius.
        plot_member(
            features,
            targets,
            i,
            nn_dir / f"member_{i}_WI_vs_radius",
            comparison_param="layer",
            fixed_param_index=10,  # Plot for time step 10.
            radius_index=1,
            permeability_index=1,
            model=model,
            nn_dirname=nn_dir,
            trainspecs=trainspecs,
        )
        # Plot NN WI and data WI vs time.
        plot_member(
            features,
            targets,
            i,
            nn_dir / f"member_{i}_WI_vs_time",
            x_param="time",
            comparison_param="layer",
            final_time=runspecs_ensemble["constants"]["INJECTION_TIME"],
            fixed_param_index=5,  # Plot for radius 5.
            radius_index=1,
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
            r"$WI_{geo} [~]$",
            r"$V_{tot} [\mathrm{m}^3]$",
        ],
    )

# Integrate into OPM.
if True:
    integration.recompile_flow(
        nn_dir / "scalings.csv",
        runspecs_integration_1["constants"]["OPM"],
        dirname / "standardwell_impl.mako",
        dirname / "standardwell.hpp",
    )
    # integration.run_integration(
    #     runspecs_integration_1,
    #     integration_dir_1,
    #     dirname / "integration.mako",
    # )
    integration.run_integration(
        runspecs_integration_2,
        integration_dir_2,
        dirname / "integration.mako",
    )
    integration.run_integration(
        runspecs_integration_3,
        integration_dir_3,
        dirname / "integration.mako",
    )

# Plot results.
if True:
    for integration_dir in [integration_dir_1, integration_dir_2, integration_dir_3]:
        summary_files: list[pathlib.Path] = [
            integration_dir / "run_6" / "output" / "5X5M_PEACEMAN.SMSPEC",
            integration_dir / "run_0" / "output" / "100X100M_NN.SMSPEC",
            integration_dir / "run_2" / "output" / "52X52M_NN.SMSPEC",
            integration_dir / "run_4" / "output" / "27X27M_NN.SMSPEC",
            integration_dir / "run_1" / "output" / "100X100M_PEACEMAN.SMSPEC",
            integration_dir / "run_3" / "output" / "52X52M_PEACEMAN.SMSPEC",
            integration_dir / "run_5" / "output" / "27X27M_PEACEMAN.SMSPEC",
        ]
        labels: list[str] = [
            "Fine-scale benchmark",
            "100x100m NN",
            "52x52m NN",
            "27x27m NN",
            "100x100m Peaceman",
            "52x52m Peaceman",
            "27x27m Peaceman",
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
