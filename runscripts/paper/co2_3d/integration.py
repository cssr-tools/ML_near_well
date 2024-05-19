import pathlib
from typing import Any

from pyopmnearwell.ml import integration
from runspecs import runspecs_integration_1 as runspecs_integration
from runspecs import trainspecs_1, trainspecs_2

if runspecs_integration["nn_name"] == "trainspecs_1":
    trainspecs: dict[str, Any] = trainspecs_1
elif runspecs_integration["nn_name"] == "trainspecs_2":
    trainspecs: dict[str, Any] = trainspecs_2


dirname: pathlib.Path = pathlib.Path(__file__).parent


def main():
    nn_dirname: pathlib.Path = (
        dirname
        / f"nn_{runspecs_integration['ensemble_name']}_{runspecs_integration['nn_name']}"
    )
    integration_dirname: pathlib.Path = (
        dirname
        / f"{runspecs_integration['name']}_{runspecs_integration['ensemble_name']}_{runspecs_integration['nn_name']}"
    )
    integration_dirname.mkdir(parents=True, exist_ok=True)

    runspecs_integration["variables"].update(
        {
            "ML_MODEL_PATH": [
                "",
                nn_dirname / "WI.model",
                nn_dirname / "WI.model",
                nn_dirname / "WI.model",
                nn_dirname / "WI.model",
                nn_dirname / "WI.model",
                "",
                "",
                "",
                "",
                "",
            ]
        }
    )

    if trainspecs["permeability_log"] and trainspecs["WI_log"]:
        if runspecs_integration["PI_unit"] == "mD":
            template_name: str = "co2_local_stencil_extended_log_mD"
        elif runspecs_integration["PI_unit"] == "m^2":
            template_name = "co2_local_stencil_extended_log_mD_PI_m2"
        hpp_name = "local_stencil_extended_log"
    elif not trainspecs["permeability_log"] and not trainspecs["WI_log"]:
        if runspecs_integration["PI_unit"] == "mD":
            template_name = "co2_local_stencil_extended_mD"
        elif runspecs_integration["PI_unit"] == "m^2":
            template_name = "co2_local_stencil_extended_mD_PI_m2"
        hpp_name = "local_stencil_extended"
    else:
        raise ValueError(
            "No mako available for given combination of"
            + " trainspecs['permeability_log'] and trainspecs['WI_log']"
        )

    integration.recompile_flow(
        nn_dirname / "scalings.csv",
        runspecs_integration["constants"]["OPM_ML"],
        StandardWell_impl_template=template_name,
        StandardWell_template=hpp_name,
        stencil_size=3,
        local_feature_names=["pressure", "saturation", "permeability"],
    )
    integration.run_integration(
        runspecs_integration, integration_dirname, dirname / "integration.mako"
    )


if __name__ == "__main__":
    main()
