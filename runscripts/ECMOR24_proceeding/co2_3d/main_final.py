from __future__ import annotations

import json
import pathlib
import sys

import seaborn as sns
from pyopmnearwell.ml import integration, utils
from runspecs_final import (
    runspecs_integration_3D_and_Peaceman_1,
    runspecs_integration_3D_and_Peaceman_2,
    runspecs_integration_3D_and_Peaceman_3,
)

dirname: pathlib.Path = pathlib.Path(__file__).parent

sys.path.append(str(dirname / ".."))

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

ensemble_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
data_stencil_dir.mkdir(parents=True, exist_ok=True)
nn_dir.mkdir(parents=True, exist_ok=True)
for integration_dir in [
    integration_3d_dir_1,
    integration_3d_dir_2,
    integration_3d_dir_3,
]:
    integration_dir.mkdir(parents=True, exist_ok=True)


# Integrate into OPM.
if True:
    for integration_dir, runspecs_integration in zip(
        [
            integration_3d_dir_1,
            integration_3d_dir_2,
            integration_3d_dir_3,
        ],
        [
            runspecs_integration_3D_and_Peaceman_1,
            runspecs_integration_3D_and_Peaceman_2,
            runspecs_integration_3D_and_Peaceman_3,
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
                "time_window": 160.0,
                "injection_rate_per_day": float(
                    runspecs_integration_3D_and_Peaceman_1["constants"][
                        "INJECTION_RATE"
                    ]
                ),
                "first_injection_length": float(
                    runspecs_integration_3D_and_Peaceman_1["constants"]["INJ1_DAYS"]
                ),
                "first_break_length": float(
                    runspecs_integration_3D_and_Peaceman_1["constants"]["SHUT_DAYS"]
                ),
                "second_injection_length": float(
                    160
                    - runspecs_integration_3D_and_Peaceman_1["constants"]["INJ1_DAYS"]
                    - runspecs_integration_3D_and_Peaceman_1["constants"]["SHUT_DAYS"]
                ),
            }
            utils.recursive_dict_update(config, update)
            json.dump(config, f, indent=4)

        integration.run_integration(
            runspecs_integration,
            integration_dir,
            dirname / "integration.mako",
        )
