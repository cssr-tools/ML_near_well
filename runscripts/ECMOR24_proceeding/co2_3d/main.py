from __future__ import annotations

import json
import math
import pathlib
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nn import FEATURE_TO_INDEX, restructure_data
from pyopmnearwell.ml import analysis, ensemble, integration, utils
from pyopmnearwell.utils import units
from runspecs import (
    runspecs_ensemble,
    runspecs_integration_3D_and_Peaceman_1,
    runspecs_integration_3D_and_Peaceman_2,
    runspecs_integration_3D_and_Peaceman_3,
    runspecs_integration_3D_and_Peaceman_4,
    trainspecs,
)
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from upscale import CO2_3D_upscaler

dirname: pathlib.Path = pathlib.Path(__file__).parent

sys.path.append(str(dirname / ".."))
from utilstime import (
    full_ensemble,
    plot_member,
    reload_data,
    tune_and_train,
)

# Set seaborn style.
sns.set_theme(context="paper", style="whitegrid")

SEED: int = 19123
utils.enable_determinism(SEED)

# Structure directories.
ensemble_dir: pathlib.Path = dirname / "ensemble_run"
data_dir: pathlib.Path = dirname / "dataset"
data_stencil_dir: pathlib.Path = dirname / "dataset_stencil"
nn_dir: pathlib.Path = dirname / "nn"

integration_3d_dir_1: pathlib.Path = (
    dirname / runspecs_integration_3D_and_Peaceman_1["name"]
)
integration_3d_dir_2: pathlib.Path = (
    dirname / runspecs_integration_3D_and_Peaceman_2["name"]
)

integration_3d_dir_3: pathlib.Path = (
    dirname / runspecs_integration_3D_and_Peaceman_3["name"]
)
integration_3d_dir_4: pathlib.Path = (
    dirname / runspecs_integration_3D_and_Peaceman_4["name"]
)

ensemble_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
data_stencil_dir.mkdir(parents=True, exist_ok=True)
nn_dir.mkdir(parents=True, exist_ok=True)
for integration_dir in [
    integration_3d_dir_1,
    integration_3d_dir_2,
    integration_3d_dir_3,
    integration_3d_dir_4,
]:
    integration_dir.mkdir(parents=True, exist_ok=True)

# Angle between both sides of the triangle grid.
ANGLE: float = math.pi / 3


#
# # Run ensemble and extract data.
if False:
    print("=== BEFORE full_ensemble ===", flush=True)
    extracted_data: np.ndarray = full_ensemble(
        runspecs_ensemble,
        ensemble_dir,
        ecl_keywords=["PRESSURE", "SGAS", "FLOGASI+"],
        summary_keywords=["FGIT", "WGIR:INJ0"],
        keyword_scalings={
            "PRESSURE": units.BAR_TO_PASCAL,
        },
        seed=SEED,
        save_intermediate_data=True,
        intermediate_data_dir=ensemble_dir / "intermediate_data",
        # keep_result_files=True,
    )
    print("=== AFTER full_ensemble ===", flush=True)
    print(
        f"extracted_data shape={extracted_data.shape} dtype={extracted_data.dtype}",
        flush=True,
    )
    batch_size = 10  # Juster batch-størrelse etter hvor mye RAM du har
    num_members = extracted_data.shape[0]
    print("=== BEFORE HDF5 save ===", flush=True)
    with h5py.File(str(ensemble_dir / "features.h5"), "w") as f:
        dset = f.create_dataset(
            "features",
            shape=extracted_data.shape,
            dtype=extracted_data.dtype,
            compression="gzip",
        )
        for start in range(0, num_members, batch_size):
            end = min(start + batch_size, num_members)
            dset[start:end] = extracted_data[start:end]
            print(f"Saved batch {start}-{end}", flush=True)
    print("=== AFTER h5py save(features) ===", flush=True)
    # Les fra HDF5 i stedet for np.load

    print("=== BEFORE HDF5 load ===", flush=True)

    with h5py.File(str(ensemble_dir / "features.h5"), "r") as f:
        extracted_data = f["features"][
            :
        ]  # eller bruk f["features"] direkte hvis du vil ha "mmap"-lignende tilgang
    print("=== AFTER HDF5 load ===", flush=True)

    print("=== BEFORE upscaler.create_ds ===", flush=True)
    upscaler: CO2_3D_upscaler = CO2_3D_upscaler(
        extracted_data, runspecs_ensemble, data_dim=5, angle=ANGLE
    )
    print("\n=== STAGE: upscaler.create_ds START ===", flush=True)
    features, targets = upscaler.create_ds(
        ensemble_dir, step_size_x=12, step_size_t=2, keep_xcells=142
    )
    # Fjern de to innerste punktene nær brønnen
    features = features[..., 2:, :]
    targets = targets[..., 2:]

    print(
        f"create_ds shapes: features={features.shape}, targets={targets.shape}",
        flush=True,
    )
    print(
        f"create_ds dtypes: features={features.dtype}, targets={targets.dtype}",
        flush=True,
    )
    print("=== STAGE: upscaler.create_ds DONE ===", flush=True)
    print("\n=== STAGE: store_dataset(raw) START ===", flush=True)
    ensemble.store_dataset(features, targets, data_dir)
    print("=== STAGE: store_dataset(raw) DONE ===", flush=True)
    print("\n=== STAGE: restructure_data START ===", flush=True)

    restructure_data(data_dir, data_stencil_dir, trainspecs, stencil_size=3)
    print("=== STAGE: restructure_data DONE ===", flush=True)

# Plot some WIs.
if False:
    features, targets = reload_data(
        runspecs_ensemble,
        trainspecs,
        data_stencil_dir,
        # A lot of the outer cells got disregarded during upscaling, because the
        # saturation could not be fully upscaled. -> Only 11 x values.
        num_xvalues=10,
        step_size_t=1,
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
            y_param="WI_log",
        )

print("\n=== STAGE: tune_and_train START ===", flush=True)


history_csv = nn_dir / "training_history.csv"

callbacks = [
    CSVLogger(str(history_csv), append=False),
    EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=15,
        min_delta=1e-6,
        min_lr=1e-6,
        verbose=1,
    ),
]
# Tune and train model.
if False:
    original_fit = keras.Model.fit

    def fit_with_callbacks(self, *args, **kwargs):
        existing_callbacks = kwargs.get("callbacks", [])
        if existing_callbacks is None:
            existing_callbacks = []

        kwargs["callbacks"] = list(existing_callbacks) + callbacks

        return original_fit(self, *args, **kwargs)

    keras.Model.fit = fit_with_callbacks

    try:
        tune_and_train(
            trainspecs,
            data_stencil_dir,
            nn_dir,
            max_trials=10,
            lr=2e-3,
            lr_tune=1e-4,
            epochs=1000,
            bs=512,
            executions_per_trial=1,
        )
    finally:
        keras.Model.fit = original_fit

    print(f"training_history exists: {history_csv.exists()}", flush=True)
    print(f"training_history path: {history_csv}", flush=True)

    with (nn_dir / "MLNearWellConfig.json").open(
        "r", newline="", encoding="utf-8"
    ) as f:
        config = json.load(f)
        config["model_type"] = "co2_3d_time"
        json.dump(
            config,
            (nn_dir / "MLNearWellConfig.json").open("w", encoding="utf-8"),
            indent=4,
        )


print("=== STAGE: tune_and_train DONE ===", flush=True)
# Do some plotting of results and sensitivity analysis.
if False:
    model: keras.Model = keras.models.load_model(nn_dir / "bestmodel.keras")  # type: ignore
    features, targets = reload_data(
        runspecs_ensemble,
        trainspecs,
        data_stencil_dir,
        # A lot of the outer cells got disregarded during upscaling, because the
        # saturation could not be fully upscaled. -> Only 11 x values.
        num_xvalues=10,
        step_size_t=1,
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
    for integration_dir, runspecs_integration in zip(
        [
            integration_3d_dir_1,
            # integration_3d_dir_2,
            # integration_3d_dir_3,
            # integration_3d_dir_4,
        ],
        [
            runspecs_integration_3D_and_Peaceman_1,
            # runspecs_integration_3D_and_Peaceman_2,
            # runspecs_integration_3D_and_Peaceman_3,
            # runspecs_integration_3D_and_Peaceman_4,
        ],
    ):
        # Write injection schedule to config.
        config_file = dirname / "dorthe_model" / "MLNearWellConfig.json"
        if not config_file.exists() or config_file.stat().st_size == 0:
            config = {}
        else:
            with config_file.open("r", encoding="utf-8") as f:
                config = json.load(f)

        with config_file.open("w", encoding="utf-8") as f:
            update = {
                "stencil_size": 3,
                "time_window": 180,
                "injection_rate_per_day": runspecs_integration_3D_and_Peaceman_1[
                    "constants"
                ]["INJECTION_RATE"],
                "first_injection_length": runspecs_integration_3D_and_Peaceman_1[
                    "constants"
                ]["INJ1_DAYS"],
                "first_break_length": runspecs_integration_3D_and_Peaceman_1[
                    "constants"
                ]["SHUT_DAYS"],
                "second_injection_length": 180
                - runspecs_integration_3D_and_Peaceman_1["constants"]["INJ1_DAYS"]
                - runspecs_integration_3D_and_Peaceman_1["constants"]["SHUT_DAYS"],
            }
            utils.recursive_dict_update(config, update)
            json.dump(config, f, indent=4)

        integration.run_integration(
            runspecs_integration,
            integration_dir,
            dirname / "integration.mako",
        )

# Plot results.
if False:
    from ecl.summary import EclSum

    print("\n=== STAGE: plot_results START ===", flush=True)

    for savedir_3d in [
        integration_3d_dir_1,
        integration_3d_dir_2,
        integration_3d_dir_3,
        integration_3d_dir_4,
    ]:
        print(f"\n--- Plotting {savedir_3d} ---", flush=True)

        labels = [
            "Fine-scale benchmark",
            "90x90m Peaceman",
            "52x52m Peaceman",
            "27x27m Peaceman",
        ]

        summary_files = [
            savedir_3d / "run_0" / "output" / "8X8M_PEACEMAN_MORE_ZCELLS.SMSPEC",
            savedir_3d / "run_1" / "output" / "90X90M_PEACEMAN.SMSPEC",
            savedir_3d / "run_2" / "output" / "52X52M_PEACEMAN.SMSPEC",
            savedir_3d / "run_3" / "output" / "27X27M_PEACEMAN.SMSPEC",
        ]

        fig, ax = plt.subplots()

        for summary_file, label in zip(summary_files, labels):
            print(f"Reading {summary_file}", flush=True)

            summary = EclSum(str(summary_file))

            time = np.array(summary.get_values("TIME", report_only=True))
            bhp = np.array(summary.get_values("WBHP:INJ0", report_only=True))

            print(
                f"{label}: TIME={time.shape}, WBHP={bhp.shape}",
                flush=True,
            )

            linewidth = 1.5 if label.startswith("Fine-scale") else 3.0
            linestyle = "solid" if label.startswith("Fine-scale") else "dotted"

            ax.plot(
                time,
                bhp,
                label=label,
                linestyle=linestyle,
                linewidth=linewidth,
            )

        ax.set_xlabel("Time since injection start (days)")
        ax.set_ylabel("Bottom hole pressure")
        ax.legend()

        fig.tight_layout()

        outpath = savedir_3d / "bhp.svg"
        print(f"Saving {outpath}", flush=True)

        fig.savefig(outpath)
        plt.close(fig)

    print("\n=== STAGE: plot_results DONE ===", flush=True)
