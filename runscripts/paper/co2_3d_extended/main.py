import math
import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from nn import FEATURE_TO_INDEX, restructure_data
from pyopmnearwell.ml import analysis, ensemble, integration, nn, utils
from pyopmnearwell.utils import formulas, units
from runspecs import (
    runspecs_ensemble,
    runspecs_integration_2D_1,
    runspecs_integration_2D_2,
    runspecs_integration_2D_3,
    runspecs_integration_2D_4,
    runspecs_integration_3D_and_Peaceman_1,
    runspecs_integration_3D_and_Peaceman_2,
    runspecs_integration_3D_and_Peaceman_3,
    runspecs_integration_3D_and_Peaceman_4,
    trainspecs,
)
from tensorflow import keras
from upscale import CO2_3D_upscaler

dirname: pathlib.Path = pathlib.Path(__file__).parent

sys.path.append(str(dirname / ".."))
from utils import (
    full_ensemble,
    plot_member,
    read_and_plot_bhp,
    reload_data,
    tune_and_train,
)

# Set seaborn style.
sns.set_theme(context="paper", style="whitegrid")

SEED: int = 19123
utils.enable_determinism(SEED)

# Structure directories.
ensemble_dir: pathlib.Path = dirname / "ensemble"
data_dir: pathlib.Path = dirname / "dataset"
data_stencil_dir: pathlib.Path = dirname / "dataset_stencil"
nn_dir: pathlib.Path = dirname / "nn"
integration_3d_dir_1: pathlib.Path = dirname / "integration_3D_and_Peaceman_1"
integration_2d_dir_1: pathlib.Path = dirname / "integration_2D_1"
integration_3d_dir_2: pathlib.Path = dirname / "integration_3D_and_Peaceman_2"
integration_2d_dir_2: pathlib.Path = dirname / "integration_2D_2"
integration_3d_dir_3: pathlib.Path = dirname / "integration_3D_and_Peaceman_3"
integration_2d_dir_3: pathlib.Path = dirname / "integration_2D_3"
integration_3d_dir_4: pathlib.Path = dirname / "integration_3D_and_Peaceman_4"
integration_2d_dir_4: pathlib.Path = dirname / "integration_2D_4"

ensemble_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
data_stencil_dir.mkdir(parents=True, exist_ok=True)
nn_dir.mkdir(parents=True, exist_ok=True)
integration_3d_dir_1.mkdir(parents=True, exist_ok=True)
integration_2d_dir_1.mkdir(parents=True, exist_ok=True)
integration_3d_dir_2.mkdir(parents=True, exist_ok=True)
integration_2d_dir_2.mkdir(parents=True, exist_ok=True)
integration_3d_dir_3.mkdir(parents=True, exist_ok=True)
integration_2d_dir_3.mkdir(parents=True, exist_ok=True)
integration_3d_dir_4.mkdir(parents=True, exist_ok=True)
integration_2d_dir_4.mkdir(parents=True, exist_ok=True)

# Run ensemble and extract data.
if False:
    extracted_data: np.ndarray = full_ensemble(
        runspecs_ensemble,
        ensemble_dir,
        ecl_keywords=["PRESSURE", "SGAS", "FLOGASI+"],
        init_keywords=["PERMX", "DZ"],
        summary_keywords=["FGIT"],
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
    upscaler = CO2_3D_upscaler(extracted_data, runspecs_ensemble, 6)
    features, targets = upscaler.create_ds(ensemble_dir, step_size_x=3, step_size_t=3)
    # Glue together data from several layers to create a dataset to train a 3 layer
    # stencil network.
    ensemble.store_dataset(features, targets, data_dir)
    restructure_data(data_dir, data_stencil_dir, trainspecs, stencil_size=3)

# Plot some WIs.
if False:
    features, targets = reload_data(
        runspecs_ensemble,
        trainspecs,
        data_stencil_dir,
        # A lot of the outer cells got disregarded during upscaling, because the
        # saturation could not be fully upscaled. -> Only 5 x values.
        num_xvalues=6,
        step_size_t=3,
    )
    for i in range(0, runspecs_ensemble["npoints"], 100):
        # Plot data WI vs radius.
        plot_member(
            features,
            targets,
            i,
            data_stencil_dir / f"member_{i}_WI_vs_radius",
            comparison_param="layer",
            fixed_param_index=10,  # Plot for time step 10.
            radius_index=FEATURE_TO_INDEX["radius"],
            permeability_index=FEATURE_TO_INDEX["permeability"],
        )
        # Plot data WI vs time.
        plot_member(
            features,
            targets,
            i,
            data_stencil_dir / f"member_{i}_WI_vs_time",
            x_param="time",
            comparison_param="layer",
            final_time=runspecs_ensemble["constants"]["INJECTION_TIME"],
            fixed_param_index=3,  # Plot for radius 3.
            radius_index=FEATURE_TO_INDEX["radius"],
            permeability_index=FEATURE_TO_INDEX["permeability"],
        )

# Train model and do some plotting of results and analysis.
if False:
    tune_and_train(
        trainspecs,
        data_stencil_dir,
        nn_dir,
        max_trials=15,
        lr=1e-4,
        epochs=5000,
        executions_per_trial=1,
    )
    model: keras.Model = keras.models.load_model(nn_dir / "bestmodel.keras")  # type: ignore
    features, targets = reload_data(
        runspecs_ensemble,
        trainspecs,
        data_stencil_dir,
        # A lot of the outer cells got disregarded during upscaling, because the
        # saturation could not be fully upscaled. -> Only 5 x values.
        num_xvalues=6,
        step_size_t=3,
    )
    for i in range(0, runspecs_ensemble["npoints"], 30):
        # Plot NN WI and data WI vs radius.
        plot_member(
            features,
            targets,
            i,
            nn_dir / f"member_{i}_WI_vs_radius",
            comparison_param="layer",
            fixed_param_index=10,  # Plot for time step 10.
            radius_index=FEATURE_TO_INDEX["radius"],
            permeability_index=FEATURE_TO_INDEX["permeability"],
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
            fixed_param_index=3,  # Plot for radius 3.
            radius_index=FEATURE_TO_INDEX["radius"],
            permeability_index=FEATURE_TO_INDEX["permeability"],
            model=model,
            nn_dirname=nn_dir,
            trainspecs=trainspecs,
        )
    outputs, inputs = analysis.sensitivity_analysis(model)
    analysis.plot_analysis(
        outputs,
        inputs,
        nn_dir / "sensitivity_analysis",
        # TODO: Fix feature names
        feature_names=trainspecs["features"],
    )

# Integrate into OPM.
if False:
    integration.recompile_flow(
        nn_dir / "scalings.csv",
        runspecs_integration_3D_and_Peaceman_1["constants"]["OPM"],
        dirname / "standardwell_impl_3d.mako",
        dirname / "standardwell.hpp",
        local_feature_names=["pressure", "saturation", "permeability"],
    )
    # integration.run_integration(
    #     runspecs_integration_3D_and_Peaceman_1,
    #     integration_3d_dir_1,
    #     dirname / "integration.mako",
    # )
    # integration.run_integration(
    #     runspecs_integration_3D_and_Peaceman_2,
    #     integration_3d_dir_2,
    #     dirname / "integration.mako",
    # )
    # integration.run_integration(
    #     runspecs_integration_3D_and_Peaceman_3,
    #     integration_3d_dir_3,
    #     dirname / "integration.mako",
    # )
    integration.run_integration(
        runspecs_integration_3D_and_Peaceman_4,
        integration_3d_dir_4,
        dirname / "integration.mako",
    )
    # # Run the 2D ML near-well model for comparison.
    integration.recompile_flow(
        dirname / ".." / "co2_2d_extended_2" / "nn" / "scalings.csv",
        runspecs_integration_2D_1["constants"]["OPM"],
        dirname / "standardwell_impl_2d.mako",
        dirname / "standardwell.hpp",
    )
    # integration.run_integration(
    #     runspecs_integration_2D_1,
    #     integration_2d_dir_1,
    #     dirname / "integration.mako",
    # )
    # integration.run_integration(
    #     runspecs_integration_2D_2,
    #     integration_2d_dir_2,
    #     dirname / "integration.mako",
    # )
    # integration.run_integration(
    #     runspecs_integration_2D_3,
    #     integration_2d_dir_3,
    #     dirname / "integration.mako",
    # )
    integration.run_integration(
        runspecs_integration_2D_4,
        integration_2d_dir_4,
        dirname / "integration.mako",
    )


# Plot results.
if True:
    for j, (savedir_3d, savedir_2d) in enumerate(
        zip(
            [
                integration_3d_dir_1,
                integration_3d_dir_2,
                integration_3d_dir_3,
                integration_3d_dir_4,
            ],
            [
                integration_2d_dir_1,
                integration_2d_dir_2,
                integration_2d_dir_3,
                integration_2d_dir_4,
            ],
        )
    ):
        labels: list[str] = [
            # "Fine-scale benchmark",
            "90x90m NN 3D",
            "52x52m NN 3D",
            "27x27m NN 3D",
            # "10x10m NN 3D",
            "90x90m Peaceman",
            "52x52m Peaceman",
            "27x27m Peaceman",
            "10x10m Peaceman",
            "90x90m NN 2D",
            # "52x52m NN 2D",
            # "27x27m NN 2D",
            # "10x10m NN 2D",
        ]
        summary_files: list[pathlib.Path] = (
            # [
            #     (
            #         dirname
            #         / "integration"
            #         / "run_0"
            #         / "output"
            #         / ("5x5m_Peaceman_more_zcells").upper()
            #     ).with_suffix(".SMSPEC"),
            # ]
            # +[
            [
                (
                    savedir_3d
                    / f"run_{i}"
                    / "output"
                    / "_".join(labels[j].split(" ")).upper()
                ).with_suffix(".SMSPEC")
                # for i in range(4)
                for i, j in zip([0, 1, 2, 4, 5, 6, 7], list(range(7)))
            ]
            + [
                (
                    savedir_2d
                    / f"run_{i}"
                    / "output"
                    / "_".join(labels[i + 7].split(" ")).upper()
                    # / "_".join(labels[i + 9].split(" ")).upper()
                ).with_suffix(".SMSPEC")
                for i in range(1)
            ]
        )
        colors: list[str] = (
            # ["black"]
            # + list(plt.cm.Blues(np.linspace(0.3, 0.7, 4)))  # type: ignore
            list(plt.cm.Blues(np.linspace(0.3, 0.7, 3)))  # type: ignore
            + list(plt.cm.Greys(np.linspace(0.3, 0.7, 4)))  # type: ignore
            # list(plt.cm.Greys(np.linspace(0.3, 0.7, 4)))  # type: ignore
            + list(plt.cm.Greens(np.linspace(0.3, 0.7, 1)))  # type: ignore
        )
        linestyles: list[str] = ["dashed"] * 7 + ["dotted"] * 1
        # linestyles: list[str] = ["solid"] + ["dashed"] * 8 + ["dotted"] * 4
        read_and_plot_bhp(
            summary_files, labels, colors, linestyles, savedir_3d / "bhp.svg"
        )