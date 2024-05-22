import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from nn import FEATURE_TO_INDEX, restructure_data
from pyopmnearwell.ml import analysis, ensemble, integration, utils
from pyopmnearwell.utils import units
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
    bhp_error,
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
for integration_dir in [
    integration_3d_dir_1,
    integration_3d_dir_2,
    integration_3d_dir_3,
    integration_3d_dir_4,
    integration_2d_dir_1,
    integration_2d_dir_2,
    integration_2d_dir_3,
    integration_2d_dir_4,
]:
    integration_dir.mkdir(parents=True, exist_ok=True)

# Run ensemble and extract data.
if True:
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
if True:
    extracted_data: np.ndarray = np.load(str(ensemble_dir / "features.npy"))
    # Manually disregard runs that produce zero pressure differences between the
    # well cell and other grid cells and thus infinite WR. For some reason these were
    # not stopped during running.
    extracted_data_per_zcell: np.ndarray = extracted_data.reshape(
        extracted_data.shape[0],
        extracted_data.shape[1],
        runspecs_ensemble["constants"]["NUM_ZCELLS"],
        -1,
        extracted_data.shape[3],
    )
    invalid_members: list[int] = list(
        set(
            np.argwhere(
                np.expand_dims(extracted_data_per_zcell[..., 0, 0], axis=-1)
                - extracted_data_per_zcell[..., 1:, 0]
                == 0
            )[..., 0]
        )
    )
    extracted_data = np.delete(extracted_data, invalid_members, axis=0)
    upscaler: CO2_3D_upscaler = CO2_3D_upscaler(extracted_data, runspecs_ensemble, 6)
    features, targets = upscaler.create_ds(ensemble_dir, step_size_x=3, step_size_t=3)

    # Some WI will be negative for some reason. Remove these.
    invalid_members = list(set(np.argwhere(targets <= 0)[..., 0]))
    features = np.delete(features, invalid_members, axis=0)
    targets = np.delete(targets, invalid_members, axis=0)

    ensemble.store_dataset(features, targets, data_dir)
    restructure_data(data_dir, data_stencil_dir, trainspecs, stencil_size=3)

# Plot some WIs.
if True:
    features, targets = reload_data(
        runspecs_ensemble,
        trainspecs,
        data_stencil_dir,
        # A lot of the outer cells got disregarded during upscaling, because the
        # saturation could not be fully upscaled. -> Only 5 x values.
        num_xvalues=8,
        step_size_t=3,
    )
    for i in range(0, features.shape[0], 20):
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
            y_param="WI_log",
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
            y_param="WI_log",
        )

# Train model and do some plotting of results and analysis.
if True:
    tune_and_train(
        trainspecs,
        data_stencil_dir,
        nn_dir,
        max_trials=5,
        lr=1e-4,
        epochs=1000,
        executions_per_trial=1,
    )

if True:
    model: keras.Model = keras.models.load_model(nn_dir / "bestmodel.keras")  # type: ignore
    features, targets = reload_data(
        runspecs_ensemble,
        trainspecs,
        data_stencil_dir,
        # A lot of the outer cells got disregarded during upscaling, because the
        # saturation could not be fully upscaled. -> Only 5 x values.
        num_xvalues=8,
        step_size_t=3,
    )
    for i in range(0, features.shape[0], 20):
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
            y_param="WI_log",
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
            y_param="WI_log",
        )
    outputs, inputs = analysis.sensitivity_analysis(model)
    analysis.plot_analysis(
        outputs,
        inputs,
        nn_dir / "sensitivity_analysis",
        feature_names=trainspecs["features"],
        legend=False,
    )

# Integrate into OPM.
if True:
    integration.recompile_flow(
        nn_dir / "scalings.csv",
        runspecs_integration_3D_and_Peaceman_1["constants"]["OPM"],
        dirname / "standardwell_impl_3d.mako",
        dirname / "standardwell.hpp",
        local_feature_names=["pressure", "saturation", "permeability"],
    )
    for integration_dir, runspecs_integration in zip(
        [
            integration_3d_dir_1,
            integration_3d_dir_2,
            integration_3d_dir_3,
            integration_3d_dir_4,
        ],
        [
            runspecs_integration_3D_and_Peaceman_1,
            runspecs_integration_3D_and_Peaceman_2,
            runspecs_integration_3D_and_Peaceman_3,
            runspecs_integration_3D_and_Peaceman_4,
        ],
    ):
        integration.run_integration(
            runspecs_integration,
            integration_dir,
            dirname / "integration.mako",
        )
    # Run the 2D ML near-well model for comparison.
    integration.recompile_flow(
        dirname / ".." / "co2_2d" / "nn" / "scalings.csv",
        runspecs_integration_2D_1["constants"]["OPM"],
        dirname / "standardwell_impl_2d.mako",
        dirname / "standardwell.hpp",
    )
    for integration_dir, runspecs_integration in zip(
        [
            integration_2d_dir_1,
            integration_2d_dir_2,
            integration_2d_dir_3,
            integration_2d_dir_4,
        ],
        [
            runspecs_integration_2D_1,
            runspecs_integration_2D_2,
            runspecs_integration_2D_3,
            runspecs_integration_2D_4,
        ],
    ):
        integration.run_integration(
            runspecs_integration,
            integration_dir,
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
            "Fine-scale benchmark",
            "90x90m NN 3D",
            "52x52m NN 3D",
            "27x27m NN 3D",
            "90x90m Peaceman",
            "52x52m Peaceman",
            "27x27m Peaceman",
            "90x90m NN 2D",
            "52x52m NN 2D",
            "27x27m NN 2D",
        ]
        summary_files: list[pathlib.Path] = (
            [
                (
                    dirname
                    / savedir_3d
                    / "run_0"
                    / "output"
                    / ("8x8m_Peaceman_more_zcells").upper()
                ).with_suffix(".SMSPEC"),
            ]
            + [
                (
                    savedir_3d
                    / f"run_{i}"
                    / "output"
                    / "_".join(labels[i].split(" ")).upper()
                ).with_suffix(".SMSPEC")
                for i in range(7)
            ]
            + [
                (
                    savedir_2d
                    / f"run_{i}"
                    / "output"
                    / "_".join(labels[i + 7].split(" ")).upper()
                ).with_suffix(".SMSPEC")
                for i in range(0)
            ]
        )
        colors: list[str] = (
            ["black"]
            + list(plt.cm.Blues(np.linspace(0.3, 0.7, 3)))  # type: ignore
            + list(plt.cm.Greys(np.linspace(0.3, 0.7, 3)))  # type: ignore
        )
        linestyles: list[str] = ["solid"] + ["dashed"] * 3 + ["dotted"] * 3
        read_and_plot_bhp(
            summary_files, labels, colors, linestyles, savedir_3d / "bhp.svg"
        )
        bhp_error(summary_files, savedir_3d / "bhp_diffs.csv", 0)
