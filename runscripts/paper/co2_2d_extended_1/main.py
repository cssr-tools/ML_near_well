import pathlib
import sys
from typing import Any

import numpy as np
import seaborn as sns
from dataset import CO2_2D_Dataset
from pyopmnearwell.ml import analysis, ensemble, integration, nn, utils
from pyopmnearwell.utils import formulas, units
from runspecs import runspecs_ensemble, runspecs_integration, trainspecs
from tensorflow import keras

dirname: pathlib.Path = pathlib.Path(__file__).parent

sys.path.append(str(dirname / ".."))
from utils import full_ensemble, plot_member, tune_and_train

# Set seaborn style.
sns.set_theme(context="paper", style="whitegrid")

SEED: int = 19123
utils.enable_determinism(SEED)

# Structure directories.
ensemble_dir: pathlib.Path = dirname / "ensemble"
data_dir: pathlib.Path = dirname / "dataset"
nn_dir: pathlib.Path = dirname / "nn"
integration_dir: pathlib.Path = dirname / "integration"

ensemble_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
nn_dir.mkdir(parents=True, exist_ok=True)
integration_dir.mkdir(parents=True, exist_ok=True)

# Get OPM installations.
# TODO: These need to be adjusted for reproducing results.
OPM: pathlib.Path = pathlib.Path("/home/peter/Documents/2023_CEMRACS/opm")
FLOW: pathlib.Path = OPM / "build" / "opm-simulators" / "bin" / "flow"
OPM_ML: pathlib.Path = pathlib.Path("/home/peter/Documents/2023_CEMRACS/opm_ml")
FLOW_ML: pathlib.Path = (
    OPM_ML / "build" / "opm-simulators" / "bin" / "flow_gaswater_dissolution_diffuse"
)


# Run ensemble and create data.
if True:
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
    )
    np.save(str(ensemble_dir / "features"), extracted_data)

    # Setup dataset for training.
    extracted_data = np.load(str(ensemble_dir / "features.npy"))
    dataset = CO2_2D_Dataset(extracted_data, runspecs_ensemble, 5)
    features, targets = dataset.create_ds(ensemble_dir, step_size_x=1, step_size_t=3)

    # Remove WI_analytical for training.
    if False:
        train_features: np.ndarray = features[..., :-1]
    # Flatten features and targets before storing.
    ensemble.store_dataset(
        np.reshape(features, (-1, 3)),
        np.reshape(targets, (-1, 1)),
        data_dir,
    )

# Plot some WIs.
if False:
    for i in range(0, runspecs_ensemble["npoints"], 50):
        # Plot data WI vs radius.
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
        # Plot data WI vs time.
        plot_member(
            features,
            targets,
            i,
            data_dir / f"member_{i}_WI_vs_time",
            x_param="time",
            comparison_param="layer",
            final_time=runspecs_ensemble["constants"]["INJECTION_TIME"],
            # TODO: What does the fixed param represent?
            fixed_param_index=10,  # Plot for time step 10. -> False
            radius_index=3,
            permeability_index=1,
        )

# Train model and do some plotting of results and analysis.
if True:
    tune_and_train(trainspecs, data_dir, nn_dir)
    model: keras.Model = keras.models.load_model(nn_dir / "bestmodel.keras")  # type: ignore
    for i in range(0, runspecs_ensemble["npoints"], 30):
        # Plot NN WI and data WI vs radius.
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
        # Plot NN WI and data WI vs time.
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
            r"$V_{tot} [\mathrm{m}^3]$",
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
        (integration_dir),
        (dirname / "integration.mako"),
    )

# Plot results.
if False:
    summary_files: list[pathlib.Path] = [
        integration_dir / "run_6" / "output" / "5X5M_PEACEMAN.SMSPEC",
        integration_dir / "run_0" / "output" / "125X125M_NN.SMSPEC",
        integration_dir / "run_2" / "output" / "62x62M_NN.SMSPEC",
        integration_dir / "run_4" / "output" / "25X25M_NN.SMSPEC",
        integration_dir / "run_1" / "output" / "125X125M_PEACEMAN.SMSPEC",
        integration_dir / "run_3" / "output" / "62x62M_PEACEMAN.SMSPEC",
        integration_dir / "run_5" / "output" / "25X25M_PEACEMAN.SMSPEC",
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
